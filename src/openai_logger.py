"""Mitmproxy addon to log OpenAI-compatible chat completion traffic."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mitmproxy import http

from src.core.config import settings

SUPPORTED_ENDPOINT_SUFFIXES = ("/chat/completions",)

INPUT_PATH = settings.data_dir / "input.jsonl"
OUTPUT_PATH = settings.data_dir / "output.jsonl"

logger = logging.getLogger(__name__)


def _is_supported_endpoint(path: str) -> bool:
    clean_path = path.split("?", 1)[0]
    return any(clean_path.endswith(suffix) for suffix in SUPPORTED_ENDPOINT_SUFFIXES)


def _read_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _is_event_stream(content_type: str) -> bool:
    return "text/event-stream" in content_type.lower()


def _parse_sse_events(body_text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in body_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if not payload_text:
            continue
        if payload_text == "[DONE]":
            break
        payload = _read_json(payload_text)
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _merge_tool_call(target: dict[str, Any], delta: dict[str, Any]) -> None:
    if "id" in delta:
        target["id"] = delta["id"]
    if "type" in delta:
        target["type"] = delta["type"]
    if "function" in delta:
        target.setdefault("function", {})
        function = target["function"]
        if "name" in delta["function"]:
            function["name"] = delta["function"]["name"]
        if "arguments" in delta["function"]:
            function["arguments"] = (
                function.get("arguments", "") + delta["function"]["arguments"]
            )


def _aggregate_streamed_response(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None

    first_event = events[0]
    response: dict[str, Any] = {
        "id": first_event.get("id"),
        "object": "chat.completion",
        "created": first_event.get("created"),
        "model": first_event.get("model"),
    }

    choices_state: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None

    for event in events:
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]

        for choice in event.get("choices", []):
            index = choice.get("index")
            if not isinstance(index, int):
                continue
            state = choices_state.setdefault(
                index,
                {
                    "index": index,
                    "role": None,
                    "content": "",
                    "tool_calls": [],
                    "finish_reason": None,
                },
            )

            if "finish_reason" in choice and choice["finish_reason"] is not None:
                state["finish_reason"] = choice["finish_reason"]

            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            if "role" in delta:
                state["role"] = delta["role"]
            if "content" in delta and delta["content"] is not None:
                state["content"] += delta["content"]

            for tool_call in delta.get("tool_calls", []) or []:
                tool_index = tool_call.get("index")
                if not isinstance(tool_index, int):
                    continue
                tools = state["tool_calls"]
                while len(tools) <= tool_index:
                    tools.append({})
                _merge_tool_call(tools[tool_index], tool_call)

    choices: list[dict[str, Any]] = []
    for index in sorted(choices_state):
        state = choices_state[index]
        message: dict[str, Any] = {
            "role": state["role"] or "assistant",
        }
        if state["content"]:
            message["content"] = state["content"]
        else:
            message["content"] = None
        if state["tool_calls"]:
            message["tool_calls"] = state["tool_calls"]

        choice_payload = {
            "index": index,
            "message": message,
            "finish_reason": state["finish_reason"],
        }
        choices.append(choice_payload)

    response["choices"] = choices
    if usage is not None:
        response["usage"] = usage

    return response


def _append_jsonl(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
    except OSError as exc:
        logger.warning("Failed to write JSONL to %s: %s", path, exc)


class OpenAILogger:
    """Log OpenAI API request/response bodies to JSONL files."""

    def request(self, flow: http.HTTPFlow) -> None:
        if flow.request.method.upper() != "POST":
            return
        if not _is_supported_endpoint(flow.request.path):
            return

        body_text = flow.request.get_text(strict=False)
        if not body_text:
            return

        payload = _read_json(body_text)
        if payload is None:
            return

        flow.metadata["openai_chat_request"] = payload
        _append_jsonl(INPUT_PATH, payload)

    def response(self, flow: http.HTTPFlow) -> None:
        if "openai_chat_request" not in flow.metadata:
            return

        body_text = flow.response.get_text(strict=False)
        if not body_text:
            return

        content_type = flow.response.headers.get("content-type", "")
        if _is_event_stream(content_type):
            events = _parse_sse_events(body_text)
            payload = _aggregate_streamed_response(events)
            if payload is None:
                return
            _append_jsonl(OUTPUT_PATH, payload)
            return

        payload = _read_json(body_text)
        if payload is None:
            return

        _append_jsonl(OUTPUT_PATH, payload)


addons = [OpenAILogger()]
