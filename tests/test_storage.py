"""Tests for log storage helpers."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from src.log_storage import StorageRouter, _safe_path_segment, append_jsonl


def test_append_jsonl_writes_single_object(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    payload = {"key": "value"}

    append_jsonl(path, payload)

    content = path.read_bytes()
    lines = content.strip().split(b"\n")
    assert len(lines) == 1
    assert orjson.loads(lines[0]) == payload


def test_append_jsonl_appends_multiple_objects(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    payloads = [
        {"id": 1, "name": "first"},
        {"id": 2, "name": "second"},
        {"id": 3, "name": "third"},
    ]

    for payload in payloads:
        append_jsonl(path, payload)

    content = path.read_bytes()
    lines = content.strip().split(b"\n")
    assert len(lines) == 3
    for index, line in enumerate(lines):
        assert orjson.loads(line) == payloads[index]


def test_append_jsonl_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "subdir" / "test.jsonl"
    payload = {"key": "value"}

    append_jsonl(path, payload)

    assert path.exists()
    content = path.read_bytes()
    lines = content.strip().split(b"\n")
    assert orjson.loads(lines[0]) == payload


def test_append_jsonl_handles_oserror(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "broken.jsonl"

    def raise_oserror(*_args: object, **_kwargs: object):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", raise_oserror)

    with caplog.at_level("WARNING"):
        append_jsonl(path, {"key": "value"})

    assert any("Failed to write JSONL" in record.message for record in caplog.records)


def test_safe_path_segment_sanitizes() -> None:
    assert _safe_path_segment("  Batch/1  ") == "Batch_1"


def test_safe_path_segment_empty_falls_back_to_unknown() -> None:
    assert _safe_path_segment("  !!!  ") == "unknown"


def test_safe_path_segment_truncates_long_values() -> None:
    value = "a" * 130
    assert len(_safe_path_segment(value)) == 120


def test_storage_router_with_header(tmp_path: Path, make_flow) -> None:
    flow = make_flow(request_headers={"x-batch-id": "Batch 1"})
    router = StorageRouter(tmp_path)

    paths = router.resolve(flow)

    base_dir = tmp_path / "requests" / "Batch_1"
    assert paths.input_path == base_dir / "input.jsonl"
    assert paths.output_path == base_dir / "output.jsonl"


def test_storage_router_without_header(tmp_path: Path, make_flow) -> None:
    flow = make_flow()
    router = StorageRouter(tmp_path)

    paths = router.resolve(flow)

    assert paths.input_path == tmp_path / "input.jsonl"
    assert paths.output_path == tmp_path / "output.jsonl"
