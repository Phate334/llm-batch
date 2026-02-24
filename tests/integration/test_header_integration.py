"""Integration validation for x-batch-id header routing output."""

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


def _write_sharegpt_dataset(dataset_path: Path) -> None:
    """Create a minimal ShareGPT-style dataset with two conversations."""

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "sharegpt-integration-1",
            "conversations": [
                {"from": "human", "value": "Give me a short greeting."},
                {"from": "assistant", "value": "Hello!"},
            ],
        },
        {
            "id": "sharegpt-integration-2",
            "conversations": [
                {"from": "human", "value": "Name two animals."},
                {"from": "assistant", "value": "Cat and dog."},
            ],
        },
    ]
    dataset_path.write_text(json.dumps(rows), encoding="utf-8")


@pytest.mark.integration  # type: ignore[misc]
def test_x_batch_id_header_routes_output_to_request_subdirectory() -> None:
    """Run benchmark with x-batch-id header and validate routed output path."""

    dataset_path = Path("data") / "ShareGPT_V3_unfiltered_cleaned_split.json"
    _write_sharegpt_dataset(dataset_path)

    command = build_bench_command(
        "--skip-chat-template",
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        f"/app/data/{dataset_path.name}",
        "--header",
        "x-batch-id=sharegpt",
    )

    run_bench_command(command)

    records = assert_and_load_jsonl(output_path(batch_id="sharegpt"))
    assert len(records) == 2, "Expected exactly 2 output records for --num-prompts=2."
    assert_chat_completion_payloads(records)
