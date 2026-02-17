"""Tests for SSE parsing and aggregation."""

from __future__ import annotations

from typing import Any

from src.openai_logger import (
    aggregate_streamed_response,
    is_event_stream,
    parse_sse_events,
)


def test_is_event_stream_is_case_insensitive() -> None:
    assert is_event_stream("Text/Event-Stream; charset=utf-8") is True


def test_parse_sse_events_stops_on_done() -> None:
    body = "\n".join(
        [
            'data: {"id": "evt_1", "choices": []}',
            "",
            "data: [DONE]",
            'data: {"id": "evt_2"}',
        ]
    )

    events = parse_sse_events(body)

    assert events == [{"id": "evt_1", "choices": []}]


def test_parse_sse_events_filters_invalid_payloads() -> None:
    body = "\n".join(
        [
            "event: ping",
            "data:",
            "data: not json",
            "",
        ]
    )

    events = parse_sse_events(body)

    assert events == []


def test_aggregate_streamed_response_empty() -> None:
    assert aggregate_streamed_response([]) is None


def test_aggregate_streamed_response_content() -> None:
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_1",
            "created": 1,
            "model": "gpt-test",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}}],
        },
        {
            "choices": [
                {"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}
            ],
        },
    ]

    response = aggregate_streamed_response(events)

    assert response is not None
    assert response["id"] == "chatcmpl_1"
    assert response["object"] == "chat.completion"
    assert response["created"] == 1
    assert response["model"] == "gpt-test"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert response["choices"][0]["message"]["content"] == "Hello"
    assert response["choices"][0]["finish_reason"] == "stop"


def test_aggregate_streamed_response_merges_tool_calls() -> None:
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_2",
            "created": 2,
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "do", "arguments": '{"a": '},
                            }
                        ]
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]
                    },
                }
            ],
        },
    ]

    response = aggregate_streamed_response(events)

    assert response is not None
    tool_calls = response["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0]["id"] == "call_1"
    assert tool_calls[0]["type"] == "function"
    assert tool_calls[0]["function"]["name"] == "do"
    assert tool_calls[0]["function"]["arguments"] == '{"a": 1}'


def test_aggregate_streamed_response_uses_last_usage() -> None:
    events: list[dict[str, Any]] = [
        {
            "id": "chatcmpl_3",
            "created": 3,
            "model": "gpt-test",
            "choices": [{"index": 0, "delta": {"content": "Hi"}}],
        },
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
        },
    ]

    response = aggregate_streamed_response(events)

    assert response is not None
    assert response["usage"] == {"total_tokens": 10}
