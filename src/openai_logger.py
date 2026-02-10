"""Mitmproxy addon to log OpenAI-compatible chat completion traffic."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mitmproxy import http

SUPPORTED_ENDPOINT_SUFFIXES = ("/chat/completions",)

DATA_DIR = Path(os.getenv("OPENAI_LOGGER_DATA_DIR", "data"))

INPUT_PATH = DATA_DIR / "input.jsonl"
OUTPUT_PATH = DATA_DIR / "output.jsonl"

logger = logging.getLogger(__name__)


def _is_supported_endpoint(path: str) -> bool:
    clean_path = path.split("?", 1)[0]
    return any(clean_path.endswith(suffix) for suffix in SUPPORTED_ENDPOINT_SUFFIXES)


def _read_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _append_jsonl(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
            handle.write("\n")
    except OSError as exc:
        logger.warning("Failed to write JSONL to %s: %s", path, exc)


class OpenAIChatCompletionsLogger:
    """Log chat completion request/response bodies to JSONL files."""

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
        # if payload.get("stream") is True:
        #     return

        flow.metadata["openai_chat_request"] = payload
        _append_jsonl(INPUT_PATH, payload)

    def response(self, flow: http.HTTPFlow) -> None:
        if "openai_chat_request" not in flow.metadata:
            return

        body_text = flow.response.get_text(strict=False)
        if not body_text:
            return

        payload = _read_json(body_text)
        if payload is None:
            return

        _append_jsonl(OUTPUT_PATH, payload)


addons = [OpenAIChatCompletionsLogger()]
