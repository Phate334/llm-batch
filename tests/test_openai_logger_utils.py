"""Tests for OpenAI logger utilities."""

from __future__ import annotations

from src.openai_logger import read_json


def test_read_json_valid_object() -> None:
    result = read_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_read_json_invalid() -> None:
    result = read_json("invalid json")
    assert result is None


def test_read_json_empty() -> None:
    result = read_json("")
    assert result is None


def test_read_json_list_payload() -> None:
    result = read_json("[1, 2, 3]")
    assert result == [1, 2, 3]
