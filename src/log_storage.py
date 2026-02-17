"""Log storage routing and JSONL writing."""

import logging
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from mitmproxy import http

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoragePaths:
    """Resolved storage paths for input and output payloads."""

    input_path: Path
    output_path: Path


def append_jsonl(path: Path, payload: Any) -> None:
    """Append a JSON payload to a JSONL file."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(orjson.dumps(payload))
            handle.write(b"\n")
    except OSError as exc:
        logger.warning("Failed to write JSONL to %s: %s", path, exc)


def _safe_path_segment(value: str) -> str:
    allowed = set(string.ascii_letters + string.digits + "-_.")
    cleaned = "".join(char if char in allowed else "_" for char in value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:120] or "unknown"


class StorageRouter:
    """Resolve storage paths based on request metadata."""

    def __init__(self, data_dir: Path, header_key: str | None = "x-batch-id") -> None:
        self._data_dir = data_dir
        self._header_key = header_key.lower() if header_key else None

    def resolve(self, flow: http.HTTPFlow) -> StoragePaths:
        """Resolve input/output paths for the given flow."""

        if self._header_key:
            header_value = flow.request.headers.get(self._header_key, "")
            if header_value:
                safe_value = _safe_path_segment(header_value)
                base_dir = self._data_dir / "requests" / safe_value
                return StoragePaths(
                    input_path=base_dir / "input.jsonl",
                    output_path=base_dir / "output.jsonl",
                )

        return StoragePaths(
            input_path=self._data_dir / "input.jsonl",
            output_path=self._data_dir / "output.jsonl",
        )
