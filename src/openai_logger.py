"""Mitmproxy addon for logging OpenAI-compatible traffic."""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import orjson
from mitmproxy import http

from src.core.config import settings
from src.log_storage import StoragePaths, StorageRouter, append_jsonl

REQUEST_METADATA_KEY = "openai_request_payload"
STORAGE_METADATA_KEY = "openai_storage_paths"
CUSTOM_ID_METADATA_KEY = "openai_batch_custom_id"

# Keys stripped from the body when writing Batch API input; streaming is
# handled locally by the SSE aggregator and must not appear in Batch files.
_BATCH_BODY_STRIP_KEYS: frozenset[str] = frozenset({"stream", "stream_options"})

# Fields used for the deterministic request fingerprint.  Model is
# intentionally excluded so that the same prompt re-sent to a different
# model produces the same custom_id for easy cross-model comparison.
_FINGERPRINT_KEYS: tuple[str, ...] = (
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "max_completion_tokens",
    "max_tokens",
    "ignore_eos",
)


@dataclass(frozen=True)
class EndpointSpec:
    """Describe an OpenAI-compatible API endpoint."""

    name: str
    suffixes: tuple[str, ...]


def _normalize_path(path: str) -> str:
    return path.split("?", 1)[0]


class EndpointRegistry:
    """Match request paths to supported OpenAI endpoint specs."""

    def __init__(self, endpoints: Iterable[EndpointSpec]) -> None:
        self._endpoints = list(endpoints)

    def match(self, path: str) -> EndpointSpec | None:
        clean_path = _normalize_path(path)
        for endpoint in self._endpoints:
            if any(clean_path.endswith(suffix) for suffix in endpoint.suffixes):
                return endpoint
        return None

    def supports(self, path: str) -> bool:
        return self.match(path) is not None


def default_endpoint_registry() -> EndpointRegistry:
    """Build the default registry with chat completions support."""

    return EndpointRegistry(
        endpoints=(
            EndpointSpec(
                name="chat.completions",
                suffixes=("/chat/completions",),
            ),
        )
    )


def read_json(text: str) -> Any | None:
    """Parse JSON text and return None on decode errors."""

    try:
        return orjson.loads(text)
    except orjson.JSONDecodeError:
        return None


def compute_custom_id(payload: dict[str, Any]) -> str:
    """Compute a deterministic custom_id from the request fingerprint.

    Only sampling-related fields are hashed; *model* is excluded so that
    identical prompts sent to different models share the same id.
    """
    fingerprint = {k: payload[k] for k in _FINGERPRINT_KEYS if k in payload}
    canonical = orjson.dumps(fingerprint, option=orjson.OPT_SORT_KEYS)
    digest = hashlib.sha256(canonical).hexdigest()[:32]
    return f"req-{digest}"


def _normalize_endpoint_url(path: str) -> str:
    """Return a canonical /v1/... Batch API endpoint URL from a raw path."""
    clean = _normalize_path(path)
    idx = clean.find("/v1/")
    if idx >= 0:
        return clean[idx:]
    if not clean.startswith("/"):
        clean = "/" + clean
    return "/v1" + clean


def is_event_stream(content_type: str) -> bool:
    """Return True when the response content type indicates SSE."""

    return "text/event-stream" in content_type.lower()


def parse_sse_events(body_text: str) -> list[dict[str, Any]]:
    """Parse SSE response text into a list of JSON event payloads."""

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
        payload = read_json(payload_text)
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


def aggregate_streamed_response(
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Aggregate streamed chat completion events into one response payload."""

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


class RequestPreprocessor(Protocol):
    """Callable signature for request preprocessors."""

    def __call__(
        self, payload: dict[str, Any], flow: http.HTTPFlow
    ) -> dict[str, Any] | None: ...


def _apply_preprocessors(
    payload: dict[str, Any],
    flow: http.HTTPFlow,
    preprocessors: Iterable[RequestPreprocessor],
) -> dict[str, Any] | None:
    for preprocessor in preprocessors:
        updated = preprocessor(payload, flow)
        if updated is None:
            return None
        payload = updated
    return payload


class OpenAILogger:
    """Log OpenAI API request/response bodies to JSONL files."""

    def __init__(
        self,
        endpoint_registry: EndpointRegistry | None = None,
        storage_router: StorageRouter | None = None,
        request_preprocessors: Iterable[RequestPreprocessor] | None = None,
    ) -> None:
        self._endpoint_registry = endpoint_registry or default_endpoint_registry()
        self._storage_router = storage_router or StorageRouter(settings.data_dir)
        self._request_preprocessors = list(request_preprocessors or [])

    def request(self, flow: http.HTTPFlow) -> None:
        if flow.request.method.upper() != "POST":
            return
        if not self._endpoint_registry.supports(flow.request.path):
            return

        body_text = flow.request.get_text(strict=False)
        if not body_text:
            return

        payload = read_json(body_text)
        if not isinstance(payload, dict):
            return

        payload = _apply_preprocessors(payload, flow, self._request_preprocessors)
        if payload is None:
            return

        custom_id = compute_custom_id(payload)
        url = _normalize_endpoint_url(flow.request.path)
        body = {k: v for k, v in payload.items() if k not in _BATCH_BODY_STRIP_KEYS}
        batch_entry: dict[str, Any] = {
            "custom_id": custom_id,
            "method": "POST",
            "url": url,
            "body": body,
        }

        storage_paths = self._storage_router.resolve(flow)
        flow.metadata[REQUEST_METADATA_KEY] = payload
        flow.metadata[CUSTOM_ID_METADATA_KEY] = custom_id
        flow.metadata[STORAGE_METADATA_KEY] = storage_paths
        append_jsonl(storage_paths.input_path, batch_entry)

    def response(self, flow: http.HTTPFlow) -> None:
        if REQUEST_METADATA_KEY not in flow.metadata:
            return
        if flow.response is None:
            return

        storage_paths = flow.metadata.get(STORAGE_METADATA_KEY)
        if not isinstance(storage_paths, StoragePaths):
            return

        body_text = flow.response.get_text(strict=False)
        if not body_text:
            return

        content_type = flow.response.headers.get("content-type", "")
        if is_event_stream(content_type):
            events = parse_sse_events(body_text)
            body_payload = aggregate_streamed_response(events)
            if body_payload is None:
                return
        else:
            body_payload = read_json(body_text)
            if body_payload is None:
                return

        custom_id = str(flow.metadata.get(CUSTOM_ID_METADATA_KEY, ""))
        status_code: int = flow.response.status_code
        request_id = flow.response.headers.get("x-request-id")

        response_data: dict[str, Any] = {
            "status_code": status_code,
            "body": body_payload,
        }
        if request_id:
            response_data["request_id"] = request_id

        batch_output: dict[str, Any] = {
            "id": f"batch_req_{custom_id}",
            "custom_id": custom_id,
            "response": response_data,
            "error": None,
        }
        append_jsonl(storage_paths.output_path, batch_output)


addons = [OpenAILogger()]
