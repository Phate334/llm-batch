"""Tests for OpenAILogger response handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.log_storage import StoragePaths
from src.openai_logger import REQUEST_METADATA_KEY, STORAGE_METADATA_KEY, OpenAILogger


def test_response_skips_without_request_metadata(
    make_flow, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow = make_flow(response_body='{"key": "value"}')
    calls: list[tuple[Path, Any]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger()
    logger.response(flow)

    assert calls == []


def test_response_skips_invalid_storage_paths(
    make_flow, monkeypatch: pytest.MonkeyPatch
) -> None:
    flow = make_flow(response_body='{"key": "value"}')
    flow.metadata[REQUEST_METADATA_KEY] = {"key": "value"}
    flow.metadata[STORAGE_METADATA_KEY] = "not paths"
    calls: list[tuple[Path, Any]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger()
    logger.response(flow)

    assert calls == []


def test_response_skips_empty_body(
    make_flow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = make_flow(response_body="")
    flow.metadata[REQUEST_METADATA_KEY] = {}
    flow.metadata[STORAGE_METADATA_KEY] = StoragePaths(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
    )
    calls: list[tuple[Path, Any]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger()
    logger.response(flow)

    assert calls == []


def test_response_skips_invalid_json(
    make_flow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = make_flow(
        response_body="not json", response_headers={"content-type": "application/json"}
    )
    flow.metadata[REQUEST_METADATA_KEY] = {}
    flow.metadata[STORAGE_METADATA_KEY] = StoragePaths(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
    )
    calls: list[tuple[Path, Any]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger()
    logger.response(flow)

    assert calls == []


def test_response_writes_json_payload(
    make_flow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = make_flow(
        response_body='{"status": "ok"}',
        response_headers={"content-type": "application/json"},
    )
    flow.metadata[REQUEST_METADATA_KEY] = {}
    paths = StoragePaths(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
    )
    flow.metadata[STORAGE_METADATA_KEY] = paths
    calls: list[tuple[Path, Any]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl",
        lambda path, payload: calls.append((path, payload)),
    )

    logger = OpenAILogger()
    logger.response(flow)

    assert calls == [(paths.output_path, {"status": "ok"})]


def test_response_writes_sse_aggregate(
    make_flow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = "\n".join(
        [
            'data: {"id": "chatcmpl_1", "created": 1, "model": "gpt", "choices": [{"index": 0, "delta": {"content": "Hi"}}]}',
            "data: [DONE]",
        ]
    )
    flow = make_flow(
        response_body=body,
        response_headers={"content-type": "text/event-stream; charset=utf-8"},
    )
    flow.metadata[REQUEST_METADATA_KEY] = {}
    paths = StoragePaths(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
    )
    flow.metadata[STORAGE_METADATA_KEY] = paths
    calls: list[tuple[Path, Any]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl",
        lambda path, payload: calls.append((path, payload)),
    )

    logger = OpenAILogger()
    logger.response(flow)

    assert len(calls) == 1
    assert calls[0][0] == paths.output_path
    assert calls[0][1]["choices"][0]["message"]["content"] == "Hi"


def test_response_skips_empty_sse(
    make_flow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    flow = make_flow(
        response_body="data: [DONE]",
        response_headers={"content-type": "text/event-stream"},
    )
    flow.metadata[REQUEST_METADATA_KEY] = {}
    flow.metadata[STORAGE_METADATA_KEY] = StoragePaths(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "output.jsonl",
    )
    calls: list[tuple[Path, Any]] = []

    monkeypatch.setattr(
        "src.openai_logger.append_jsonl", lambda *_: calls.append((Path("x"), {}))
    )

    logger = OpenAILogger()
    logger.response(flow)

    assert calls == []
