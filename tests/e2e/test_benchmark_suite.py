"""Tests for the production benchmark suite and statistical report generator."""

from pathlib import Path
import pytest

from benchmarks.bench_throughput import run_throughput_benchmark
from benchmarks.report_generator import (
    BenchmarkResult,
    compute_percentiles,
    generate_latex_table,
    generate_markdown_report,
    save_benchmark_report,
)


def test_compute_percentiles_accuracy() -> None:
    """Verify statistical percentile calculations against known distributions."""
    # 1 to 100
    values = [float(x) for x in range(1, 101)]
    p = compute_percentiles(values)

    assert p["mean"] == pytest.approx(50.5)
    assert p["p50"] == pytest.approx(50.5)
    assert p["p90"] == pytest.approx(90.1)
    assert p["p99"] == pytest.approx(99.01)

    # Empty values edge case
    empty_p = compute_percentiles([])
    assert empty_p["mean"] == 0.0
    assert empty_p["p50"] == 0.0


def test_in_process_throughput_benchmark_run() -> None:
    """Verify in-process throughput benchmark executes cleanly and produces valid metrics."""
    result = run_throughput_benchmark(
        num_requests=4,
        prompt_len=16,
        output_len=8,
        device="cpu",
    )

    assert isinstance(result, BenchmarkResult)
    assert result.num_requests == 4
    assert result.total_output_tokens >= 4
    assert result.duration_s > 0.0
    assert result.request_throughput_req_per_s > 0.0
    assert result.token_throughput_tok_per_s > 0.0
    assert result.ttft_ms["p50"] >= 0.0
    assert result.tpot_ms["p50"] >= 0.0


def test_report_generator_markdown_and_latex(tmp_path: Path) -> None:
    """Verify Markdown, LaTeX, and JSON report generation."""
    sample_result = BenchmarkResult(
        benchmark_name="test_workload",
        num_requests=100,
        total_input_tokens=5000,
        total_output_tokens=2500,
        duration_s=10.0,
        request_throughput_req_per_s=10.0,
        token_throughput_tok_per_s=250.0,
        ttft_ms={"mean": 12.0, "p50": 10.0, "p90": 18.0, "p95": 20.0, "p99": 25.0},
        tpot_ms={"mean": 4.0, "p50": 3.8, "p90": 4.5, "p95": 5.0, "p99": 6.0},
        latency_ms={"mean": 100.0, "p50": 95.0, "p90": 120.0, "p95": 130.0, "p99": 150.0},
    )

    md_report = generate_markdown_report([sample_result])
    assert "# nanoserve Serving Performance Benchmark Report" in md_report
    assert "test_workload" in md_report
    assert "250.0" in md_report

    tex_table = generate_latex_table([sample_result])
    assert r"\begin{table*}[t]" in tex_table
    assert "test\\_workload" in tex_table
    assert r"\end{table*}" in tex_table

    # File serialization
    save_benchmark_report([sample_result], tmp_path)
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_report.md").exists()
    assert (tmp_path / "benchmark_table.tex").exists()
