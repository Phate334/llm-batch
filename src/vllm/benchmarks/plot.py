# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate plots for benchmark results."""

import importlib
from pathlib import Path
from typing import Any

from vllm.utils.import_utils import PlaceholderModule

try:
    px = importlib.import_module("plotly.express")
    pio = importlib.import_module("plotly.io")
except ImportError:
    _plotly = PlaceholderModule("plotly")
    px = _plotly.placeholder_attr("express")
    pio = _plotly.placeholder_attr("io")

try:
    plt = importlib.import_module("matplotlib.pyplot")
except ImportError:
    _matplotlib = PlaceholderModule("matplotlib")
    plt = _matplotlib.placeholder_attr("pyplot")


def generate_timeline_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    colors: list[str] | None = None,
    itl_thresholds: list[float] | None = None,
    labels: list[str] | None = None,
) -> None:
    """Generate an HTML timeline plot from benchmark results."""
    if colors is None:
        colors = ["#109618", "#FF7F0E", "#D62728"]
    if itl_thresholds is None:
        itl_thresholds = [0.025, 0.050]
    if labels is None:
        labels = [
            f"ITL < {itl_thresholds[0] * 1000:.0f}ms",
            f"{itl_thresholds[0] * 1000:.0f}ms <= ITL < {itl_thresholds[1] * 1000:.0f}ms",
            f"ITL >= {itl_thresholds[1] * 1000:.0f}ms",
        ]

    labels_colors = {"TTFT": "#636EFA", **dict(zip(labels, colors))}
    labels_order = ["TTFT"] + labels

    timeline_data = construct_timeline_data(results, itl_thresholds, labels)

    if not timeline_data:
        print("No timeline data to plot")
        return

    fig = px.timeline(
        timeline_data,
        x_start="start",
        x_end="end",
        y="request_id",
        color="type",
        color_discrete_map=labels_colors,
        category_orders={"type": labels_order},
        hover_data=[
            "prompt_tokens",
            "output_tokens",
            "req_start_time",
            "req_finish_time",
            "segment_start",
            "segment_end",
            "duration",
        ],
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>"
        "Type: %{fullData.name}<br>"
        "Start: %{customdata[4]}<br>"
        "End: %{customdata[5]}<br>"
        "Duration: %{customdata[6]}<br>"
        "Prompt Tokens: %{customdata[0]}<br>"
        "Output Tokens: %{customdata[1]}<br>"
        "Request Start Time: %{customdata[2]}<br>"
        "Request End Time: %{customdata[3]}<br>"
        "<extra></extra>"
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Request ID",
        showlegend=True,
    )

    pio.write_html(fig, str(output_path))
    print(f"Timeline plot saved to: {output_path}")


def construct_timeline_data(
    requests_data: list[dict[str, Any]],
    itl_thresholds: list[float],
    labels: list[str],
) -> list[dict[str, Any]]:
    """Construct timeline data from request results."""

    def tostr(sec_time: float) -> str:
        h = int(sec_time // 3600)
        assert h < 100, "time seems to last more than 100 hours"
        m = int((sec_time % 3600) // 60)
        s = sec_time % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def itl_type(itl: float) -> str:
        if itl < itl_thresholds[0]:
            return labels[0]
        if itl < itl_thresholds[1]:
            return labels[1]
        return labels[2]

    t0 = None
    for request in requests_data:
        start_time = request.get("start_time")
        if start_time is not None and (t0 is None or start_time < t0):
            t0 = start_time

    if t0 is None:
        return []

    timeline_data = []
    for i, request in enumerate(requests_data):
        start_time = request.get("start_time")
        ttft = request.get("ttft")
        itl = request.get("itl", [])
        latency = request.get("latency")
        prompt_len = request.get("prompt_len", 0)
        output_tokens = request.get("output_tokens", 0)

        if start_time is None or ttft is None or latency is None:
            continue

        start_time = start_time - t0
        start_time_str = tostr(start_time)

        ttft_end = start_time + ttft
        ttft_end_str = tostr(ttft_end)

        timeline_data.append(
            {
                "request_id": f"Req {i}",
                "start": start_time_str,
                "end": ttft_end_str,
                "type": "TTFT",
                "prompt_tokens": prompt_len,
                "output_tokens": output_tokens,
                "req_start_time": tostr(start_time),
                "req_finish_time": tostr(start_time + latency),
                "segment_start": start_time_str,
                "segment_end": ttft_end_str,
                "duration": f"{ttft:.3f}s",
            }
        )

        prev_time = ttft_end
        prev_time_str = ttft_end_str
        for itl_value in itl:
            itl_end = prev_time + itl_value
            itl_end_str = tostr(itl_end)

            timeline_data.append(
                {
                    "request_id": f"Req {i}",
                    "start": prev_time_str,
                    "end": itl_end_str,
                    "type": itl_type(itl_value),
                    "prompt_tokens": prompt_len,
                    "output_tokens": output_tokens,
                    "req_start_time": tostr(start_time),
                    "req_finish_time": tostr(start_time + latency),
                    "segment_start": prev_time_str,
                    "segment_end": itl_end_str,
                    "duration": f"{itl_value:.3f}s",
                }
            )

            prev_time = itl_end
            prev_time_str = itl_end_str

    return timeline_data


def generate_dataset_stats_plot(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate a matplotlib figure with dataset statistics."""
    prompt_tokens = []
    output_tokens = []
    total_tokens = []

    for request in results:
        prompt_len = request.get("prompt_len", 0)
        output_len = request.get("output_tokens", 0)

        prompt_tokens.append(prompt_len)
        output_tokens.append(output_len)
        total_tokens.append(prompt_len + output_len)

    if not prompt_tokens:
        print("No data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(prompt_tokens, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Prompt Tokens Distribution")
    axes[0, 0].set_xlabel("Prompt Tokens")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(output_tokens, bins=30, edgecolor="black", alpha=0.7)
    axes[0, 1].set_title("Output Tokens Distribution")
    axes[0, 1].set_xlabel("Output Tokens")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].hist(total_tokens, bins=30, edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Total Tokens Distribution")
    axes[1, 0].set_xlabel("Total Tokens")
    axes[1, 0].set_ylabel("Count")

    request_ids = list(range(len(prompt_tokens)))
    axes[1, 1].bar(request_ids, prompt_tokens, label="Prompt")
    axes[1, 1].bar(
        request_ids,
        output_tokens,
        bottom=prompt_tokens,
        label="Output",
    )
    axes[1, 1].set_title("Prompt + Output Tokens per Request")
    axes[1, 1].set_xlabel("Request ID")
    axes[1, 1].set_ylabel("Tokens")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Dataset statistics plot saved to: {output_path}")
