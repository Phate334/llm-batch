"""Integration validation for x-batch-id header routing output."""

from pathlib import Path

import pytest

from .helpers import (
    assert_and_load_jsonl,
    assert_chat_completion_payloads,
    build_bench_command,
    output_path,
    run_bench_command,
)


@pytest.mark.integration  # type: ignore[misc]
def test_x_batch_id_header_routes_output_to_request_subdirectory() -> None:
    """Run benchmark with x-batch-id header and validate routed output path."""

    host_dataset_path = Path("data") / "ShareGPT_V3_unfiltered_cleaned_split.json"
    if not host_dataset_path.exists():
        pytest.skip(
            "ShareGPT dataset file is not available. "
            "Expected at data/ShareGPT_V3_unfiltered_cleaned_split.json."
        )

    command = build_bench_command(
        "--skip-chat-template",
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        "/app/data/ShareGPT_V3_unfiltered_cleaned_split.json",
        "--header",
        "x-batch-id=sharegpt",
    )

    run_bench_command(command)

    records = assert_and_load_jsonl(output_path(batch_id="sharegpt"))
    assert len(records) == 2, "Expected exactly 2 output records for --num-prompts=2."
    assert_chat_completion_payloads(records)
