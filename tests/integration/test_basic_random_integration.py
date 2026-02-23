"""Integration validation for basic random benchmark output."""

import pytest

from .helpers import (
    assert_and_load_jsonl,
    assert_chat_completion_payloads,
    build_bench_command,
    output_path,
    run_bench_command,
)


@pytest.mark.integration  # type: ignore[misc]
def test_basic_random_output_exists_and_non_empty() -> None:
    """Run random benchmark and validate non-empty output.jsonl."""

    command = build_bench_command("--dataset-name", "random")
    run_bench_command(command)

    records = assert_and_load_jsonl(output_path())
    assert len(records) == 2, "Expected exactly 2 output records for --num-prompts=2."
    assert_chat_completion_payloads(records)
