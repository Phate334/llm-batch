"""Tests for OpenAILogger request handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.log_storage import StoragePaths, StorageRouter
from src.openai_logger import (
    REQUEST_METADATA_KEY,
    STORAGE_METADATA_KEY,
    EndpointRegistry,
    EndpointSpec,
    OpenAILogger,
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

    assert calls == [(storage_paths.input_path, {"key": "value", "flag": True})]
    assert flow.metadata[REQUEST_METADATA_KEY] == {"key": "value", "flag": True}
    assert flow.metadata[STORAGE_METADATA_KEY] == storage_paths
    assert router.called_with is flow
