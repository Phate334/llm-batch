"""Tests for OpenAILogger request handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.log_storage import StoragePaths, StorageRouter
from src.openai_logger import (
    CUSTOM_ID_METADATA_KEY,
    REQUEST_METADATA_KEY,
    STORAGE_METADATA_KEY,
    EndpointRegistry,
    EndpointSpec,
    OpenAILogger,
    compute_custom_id,
)


class DummyStorageRouter(StorageRouter):
    def __init__(self, paths: StoragePaths) -> None:
        super().__init__(Path("."))
        self.paths = paths
        self.called_with: Any = None

    def resolve(self, flow: Any) -> StoragePaths:
        self.called_with = flow
        return self.paths


def build_registry() -> EndpointRegistry:
    return EndpointRegistry(
        endpoints=(
            EndpointSpec(name="chat.completions", suffixes=("/chat/completions",)),
        )
    )


def test_request_skips_non_post(make_flow, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = make_flow(method="GET", request_body='{"key": "value"}')
    calls: list[tuple[Path, object]] = []

    def record(path: Path, payload: object) -> None:
        calls.append((path, payload))

    monkeypatch.setattr("src.openai_logger.append_jsonl", record)

    logger = OpenAILogger(endpoint_registry=build_registry())
    logger.request(flow)

    assert calls == []
    assert REQUEST_METADATA_KEY not in flow.metadata


def test_request_skips_unsupported_path(
    make_flow, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow = make_flow(path="/other", request_body='{"key": "value"}')
    calls: list[tuple[Path, object]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger(endpoint_registry=build_registry())
    logger.request(flow)

    assert calls == []
    assert REQUEST_METADATA_KEY not in flow.metadata


def test_request_skips_empty_body(make_flow, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = make_flow(request_body="")
    calls: list[tuple[Path, object]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger(endpoint_registry=build_registry())
    logger.request(flow)

    assert calls == []


def test_request_skips_invalid_json(make_flow, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = make_flow(request_body="not json")
    calls: list[tuple[Path, object]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger(endpoint_registry=build_registry())
    logger.request(flow)

    assert calls == []


def test_request_skips_non_object_json(
    make_flow, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow = make_flow(request_body="[]")
    calls: list[tuple[Path, object]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger(endpoint_registry=build_registry())
    logger.request(flow)

    assert calls == []


def test_request_skips_when_preprocessor_returns_none(
    make_flow, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow = make_flow(request_body='{"key": "value"}')
    calls: list[tuple[Path, object]] = []

    def preprocessor(payload: dict[str, Any], flow: Any) -> dict[str, Any] | None:
        del payload, flow
        return None

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger(
        endpoint_registry=build_registry(),
        request_preprocessors=[preprocessor],
    )
    logger.request(flow)

    assert calls == []
    assert REQUEST_METADATA_KEY not in flow.metadata


def test_request_records_payload_and_metadata(
    make_flow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = make_flow(request_body='{"key": "value"}')
    calls: list[tuple[Path, object]] = []
    storage_paths = StoragePaths(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
    )
    router = DummyStorageRouter(storage_paths)

    def preprocessor(payload: dict[str, Any], flow: Any) -> dict[str, Any]:
        del flow
        payload["flag"] = True
        return payload

    def record(path: Path, payload: object) -> None:
        calls.append((path, payload))

    monkeypatch.setattr("src.openai_logger.append_jsonl", record)

    logger = OpenAILogger(
        endpoint_registry=build_registry(),
        storage_router=router,
        request_preprocessors=[preprocessor],
    )
    logger.request(flow)

    expected_payload = {"key": "value", "flag": True}
    expected_custom_id = compute_custom_id(expected_payload)
    assert calls == [
        (
            storage_paths.input_path,
            {
                "custom_id": expected_custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": expected_payload,
            },
        )
    ]
    assert flow.metadata[REQUEST_METADATA_KEY] == expected_payload
    assert flow.metadata[CUSTOM_ID_METADATA_KEY] == expected_custom_id
    assert flow.metadata[STORAGE_METADATA_KEY] == storage_paths
    assert router.called_with is flow


def test_request_strips_stream_and_stream_options(
    make_flow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Batch body must not contain stream / stream_options."""
    payload_with_stream = {
        "messages": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "max_completion_tokens": 128,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    flow = make_flow(request_body=__import__("json").dumps(payload_with_stream))
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        "src.openai_logger.append_jsonl",
        lambda path, payload: calls.append((path, payload)),
    )
    logger = OpenAILogger(
        endpoint_registry=build_registry(),
        storage_router=DummyStorageRouter(
            StoragePaths(
                input_path=tmp_path / "input.jsonl",
                output_path=tmp_path / "output.jsonl",
            )
        ),
    )
    logger.request(flow)

    assert len(calls) == 1
    written_body = calls[0][1]["body"]  # type: ignore[index]
    assert "stream" not in written_body
    assert "stream_options" not in written_body
    assert written_body["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# compute_custom_id unit tests
# ---------------------------------------------------------------------------


def test_compute_custom_id_is_stable() -> None:
    """Same messages + sampling params always produce the same id."""
    payload = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.7}
    assert compute_custom_id(payload) == compute_custom_id(payload)


def test_compute_custom_id_same_for_different_model() -> None:
    """model field is excluded from the fingerprint."""
    base = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.5}
    with_model_a = {**base, "model": "gpt-4o"}
    with_model_b = {**base, "model": "gpt-4o-mini"}
    assert compute_custom_id(with_model_a) == compute_custom_id(with_model_b)


def test_compute_custom_id_differs_for_different_messages() -> None:
    """Different messages produce different ids."""
    p1 = {"messages": [{"role": "user", "content": "hello"}]}
    p2 = {"messages": [{"role": "user", "content": "goodbye"}]}
    assert compute_custom_id(p1) != compute_custom_id(p2)


def test_compute_custom_id_differs_for_different_sampling() -> None:
    """Different sampling params produce different ids."""
    base = {"messages": [{"role": "user", "content": "hi"}]}
    p1 = {**base, "temperature": 0.0}
    p2 = {**base, "temperature": 1.0}
    assert compute_custom_id(p1) != compute_custom_id(p2)


def test_compute_custom_id_starts_with_req_prefix() -> None:
    cid = compute_custom_id({"messages": [{"role": "user", "content": "x"}]})
    assert cid.startswith("req-")
