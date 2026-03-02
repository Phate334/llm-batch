"""Integration validation for OpenAI tool calling outputs."""

from __future__ import annotations

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


def _write_tool_dataset(dataset_path: Path) -> list[dict[str, str]]:
    """Create a custom dataset with prompts that require tool calls."""

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {"prompt": "Use the get_current_weather tool to report Taipei weather in celsius.", "city": "Taipei", "unit": "celsius"},
        {"prompt": "Use the get_current_weather tool to report Dallas weather in fahrenheit.", "city": "Dallas", "unit": "fahrenheit"},
    ]
    rows = [
        {"prompt": sample["prompt"], "output_tokens": 64} for sample in samples
    ]
    content = "\n".join(json.dumps(row) for row in rows) + "\n"
    dataset_path.write_text(content, encoding="utf-8")
    return samples


@pytest.mark.integration  # type: ignore[misc]
def test_tool_calls_are_logged_with_arguments() -> None:
    """Run benchmark with tool use and validate tool call responses."""

    dataset_path = Path("data") / "integration_tool_calls.jsonl"
    samples = _write_tool_dataset(dataset_path)

    tools_body = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name, e.g. Taipei",
                            },
                            "state": {
                                "type": "string",
                                "description": "Two-letter state code when applicable",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                            },
                        },
                        "required": ["city", "unit"],
                    },
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_current_weather"},
        },
    }

    command = build_bench_command(
        "--skip-chat-template",
        "--dataset-name",
        "custom",
        "--dataset-path",
        f"/app/data/{dataset_path.name}",
        "--disable-shuffle",
        "--extra-body",
        json.dumps(tools_body),
    )

    run_bench_command(command)

    records = assert_and_load_jsonl(output_path())
    assert len(records) == 2, "Expected exactly 2 output records for --num-prompts=2."
    assert_chat_completion_payloads(records)

    for sample, record in zip(samples, records):
        choices = record.get("choices")
        assert isinstance(choices, list) and choices, "Response choices missing."
        first_choice = choices[0]
        message = first_choice.get("message", {})
        tool_calls = message.get("tool_calls")
        assert isinstance(tool_calls, list) and tool_calls, "Expected tool_calls list."

        first_call = tool_calls[0]
        function = first_call.get("function", {})
        assert function.get("name") == "get_current_weather"

        arguments = function.get("arguments")
        assert isinstance(arguments, str) and arguments, "Tool call arguments missing."
        assert sample["city"].lower() in arguments.lower()
        assert sample["unit"].lower() in arguments.lower()
