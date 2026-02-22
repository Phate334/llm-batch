"""Integration validation for basic random benchmark output."""

import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration  # type: ignore[misc]
def test_basic_random_output_exists_and_non_empty() -> None:
    """Run random benchmark and validate non-empty output.jsonl."""

    command = [
        "docker",
        "compose",
        "-f",
        "compose.yaml",
        "exec",
        "-T",
        "batch",
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
        "--dataset-name",
        "random",
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    output_path = Path("data/output.jsonl")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
