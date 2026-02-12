"""Tests for openai_logger module."""

import tempfile
from pathlib import Path

import orjson

from src.openai_logger import _append_jsonl, _read_json


class TestReadJson:
    """Test _read_json function."""

    def test_valid_json(self):
        """Test parsing valid JSON string."""
        result = _read_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_with_unicode(self):
        """Test parsing JSON with unicode characters."""
        result = _read_json('{"message": "Hello 世界"}')
        assert result == {"message": "Hello 世界"}

    def test_invalid_json(self):
        """Test parsing invalid JSON string."""
        result = _read_json("invalid json")
        assert result is None

    def test_empty_string(self):
        """Test parsing empty string."""
        result = _read_json("")
        assert result is None


class TestAppendJsonl:
    """Test _append_jsonl function."""

    def test_append_single_object(self):
        """Test appending a single JSON object to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            payload = {"key": "value"}

            _append_jsonl(path, payload)

            content = path.read_bytes()
            lines = content.strip().split(b"\n")
            assert len(lines) == 1
            assert orjson.loads(lines[0]) == payload

    def test_append_multiple_objects(self):
        """Test appending multiple JSON objects to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            payloads = [
                {"id": 1, "name": "first"},
                {"id": 2, "name": "second"},
                {"id": 3, "name": "third"},
            ]

            for payload in payloads:
                _append_jsonl(path, payload)

            content = path.read_bytes()
            lines = content.strip().split(b"\n")
            assert len(lines) == 3
            for i, line in enumerate(lines):
                assert orjson.loads(line) == payloads[i]

    def test_append_with_unicode(self):
        """Test appending JSON with unicode characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            payload = {"message": "Hello 世界", "emoji": "🌟"}

            _append_jsonl(path, payload)

            content = path.read_bytes()
            lines = content.strip().split(b"\n")
            assert len(lines) == 1
            assert orjson.loads(lines[0]) == payload

    def test_creates_parent_directory(self):
        """Test that parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "test.jsonl"
            payload = {"key": "value"}

            _append_jsonl(path, payload)

            assert path.exists()
            content = path.read_bytes()
            lines = content.strip().split(b"\n")
            assert orjson.loads(lines[0]) == payload
