"""Shared helpers for integration benchmark tests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

# docker might not be installed locally (e.g. when using podman with an alias).
# subprocess calls don't respect shell aliases, so we dynamically choose the
# executable based on what's available in PATH.  This mirrors the Github
# Actions environment where `docker` is guaranteed and keeps tests portable.
_COMPOSE_CLI = shutil.which("docker") or shutil.which("podman") or "docker"

DOCKER_COMPOSE_EXEC_PREFIX = [
    _COMPOSE_CLI,
    "compose",
    "-f",
    "compose.yaml",
    "exec",
    "-T",
    "batch",
]

VLLM_BENCH_BASE_ARGS = [
    "vllm",
    "bench",
    "serve",
    "--backend",
    "openai-chat",
    "--base-url",
    "http://llm:8080",
    "--endpoint",
    "/v1/chat/completions",
    "--model",
    "lmstudio-community/gemma-3-1B-it-qat-GGUF",
    "--tokenizer",
    "google/gemma-3-1b-it",
    "--num-prompts",
    "2",
    "--max-concurrency",
    "2",
]


def build_bench_command(*extra_args: str) -> list[str]:
    """Build a complete docker-compose benchmark command."""

    return [*DOCKER_COMPOSE_EXEC_PREFIX, *VLLM_BENCH_BASE_ARGS, *extra_args]


def run_bench_command(command: list[str]) -> None:
    """Run a benchmark command and fail with rich diagnostics on error."""

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, (
        "Command failed with non-zero exit code.\n"
        f"command={' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def output_path(batch_id: str | None = None) -> Path:
    """Return the expected output path for a benchmark run."""

    base_dir = Path("data")
    if batch_id:
        return base_dir / "requests" / batch_id / "output.jsonl"
    return base_dir / "output.jsonl"


def assert_and_load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Validate that a JSONL file exists, is non-empty, and parse each object."""

    assert path.exists(), (
        "Expected output file not found after benchmark run.\n"
        f"expected_path={path.resolve()}"
    )
    assert path.stat().st_size > 0, (
        f"Expected output file to be non-empty.\nactual_size={path.stat().st_size}"
    )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            assert isinstance(item, dict), "Each JSONL line must be a JSON object."
            records.append(item)

    assert records, "Expected at least one JSON object in output JSONL."
    return records


def assert_chat_completion_payloads(records: list[dict[str, Any]]) -> None:
    """Validate the basic schema of chat completion payloads."""

    for record in records:
        choices = record.get("choices")
        assert isinstance(choices, list) and choices, (
            "Expected non-empty 'choices' in chat completion payload."
        )
        first_choice = choices[0]
        assert isinstance(first_choice, dict), "Expected each choice to be an object."
        assert any(key in first_choice for key in ("message", "delta", "text")), (
            "Expected message content field in first choice."
        )
