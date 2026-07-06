"""Benchmark reporting utilities: statistical aggregation, Markdown tables, and LaTeX exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Consolidated metrics summary for a benchmark run."""

    benchmark_name: str
    num_requests: int
    total_input_tokens: int
    total_output_tokens: int
    duration_s: float
    request_throughput_req_per_s: float
    token_throughput_tok_per_s: float
    ttft_ms: dict[str, float]
    tpot_ms: dict[str, float]
    latency_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "num_requests": self.num_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "duration_s": round(self.duration_s, 3),
            "request_throughput_req_per_s": round(self.request_throughput_req_per_s, 2),
            "token_throughput_tok_per_s": round(self.token_throughput_tok_per_s, 2),
            "ttft_ms": {k: round(v, 2) for k, v in self.ttft_ms.items()},
            "tpot_ms": {k: round(v, 2) for k, v in self.tpot_ms.items()},
            "latency_ms": {k: round(v, 2) for k, v in self.latency_ms.items()},
        }


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """Compute statistical mean and percentiles (p50, p90, p95, p99)."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def generate_markdown_report(results: list[BenchmarkResult]) -> str:
    """Generate Markdown summary table from benchmark results."""
    lines = [
        "# nanoserve Serving Performance Benchmark Report",
        "",
        "| Benchmark | Reqs | In / Out Tokens | Throughput (tok/s) | TTFT P50 (ms) | TTFT P99 (ms) | TPOT P50 (ms) | TPOT P99 (ms) | E2E P50 (ms) | E2E P99 (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        lines.append(
            f"| {r.benchmark_name} | {r.num_requests} | {r.total_input_tokens} / {r.total_output_tokens} | "
            f"{r.token_throughput_tok_per_s:.1f} | "
            f"{r.ttft_ms.get('p50', 0.0):.1f} | {r.ttft_ms.get('p99', 0.0):.1f} | "
            f"{r.tpot_ms.get('p50', 0.0):.1f} | {r.tpot_ms.get('p99', 0.0):.1f} | "
            f"{r.latency_ms.get('p50', 0.0):.1f} | {r.latency_ms.get('p99', 0.0):.1f} |"
        )

    return "\n".join(lines)


def generate_latex_table(results: list[BenchmarkResult]) -> str:
    """Generate publication-ready LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Workload} & \textbf{Throughput (tok/s)} & \textbf{TTFT P50 (ms)} & \textbf{TTFT P99 (ms)} & \textbf{TPOT P50 (ms)} & \textbf{TPOT P99 (ms)} & \textbf{Latency P99 (ms)} \\",
        r"\midrule",
    ]

    for r in results:
        escaped_name = r.benchmark_name.replace("_", r"\_")
        lines.append(
            f"{escaped_name} & {r.token_throughput_tok_per_s:.1f} & "
            f"{r.ttft_ms.get('p50', 0.0):.1f} & {r.ttft_ms.get('p99', 0.0):.1f} & "
            f"{r.tpot_ms.get('p50', 0.0):.1f} & {r.tpot_ms.get('p99', 0.0):.1f} & "
            f"{r.latency_ms.get('p99', 0.0):.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{nanoserve high-throughput serving and latency percentiles.}",
        r"\label{tab:nanoserve_perf}",
        r"\end{table*}",
    ])

    return "\n".join(lines)


def save_benchmark_report(results: list[BenchmarkResult], output_dir: Path | str) -> None:
    """Save benchmark outputs to JSON, Markdown, and LaTeX files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_data = [r.to_dict() for r in results]
    with open(out / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    with open(out / "benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(results))

    with open(out / "benchmark_table.tex", "w", encoding="utf-8") as f:
        f.write(generate_latex_table(results))
