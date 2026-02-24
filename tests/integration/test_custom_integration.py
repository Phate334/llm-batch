"""Integration validation for custom dataset benchmark output."""

import json
from pathlib import Path

import pytest

from .helpers import (
    assert_and_load_jsonl,
    assert_chat_completion_payloads,
    build_bench_command,
    output_path,
    run_bench_command,
)


def _write_custom_dataset(dataset_path: Path) -> None:
    """Create a minimal valid custom JSONL dataset with two prompts."""

    rows = [
        {"prompt": "Write a one-line greeting.", "output_tokens": 16},
        {"prompt": "List two colors.", "output_tokens": 16},
    ]
    content = "\n".join(json.dumps(row) for row in rows) + "\n"
    dataset_path.write_text(content, encoding="utf-8")


@pytest.mark.integration  # type: ignore[misc]
def test_custom_dataset_output_exists_and_matches_prompt_count() -> None:
    """Run benchmark with a generated custom dataset and validate output JSONL."""

    dataset_path = Path("data") / "integration_custom_prompts.jsonl"
    _write_custom_dataset(dataset_path)

    command = build_bench_command(
        "--skip-chat-template",
        "--dataset-name",
        "custom",
        "--dataset-path",
        f"/app/data/{dataset_path.name}",
    )

    run_bench_command(command)

    records = assert_and_load_jsonl(output_path())
    assert len(records) == 2, "Expected exactly 2 output records for --num-prompts=2."
    assert_chat_completion_payloads(records)
