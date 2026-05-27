"""Benchmark harness: fixed-concurrency sweeps with per-request timing."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class RequestResult:
    """Timing for a single benchmark request."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    e2e_ms: float
    tpot_ms: float  # average time per output token


@dataclass
class SweepResult:
    """Results for a single concurrency level."""

    concurrency: int
    num_requests: int
    total_time_s: float
    throughput_tok_s: float
    results: list[RequestResult] = field(default_factory=list)

    @property
    def p50_ttft_ms(self) -> float:
        ttfts = sorted(r.ttft_ms for r in self.results)
        return ttfts[len(ttfts) // 2] if ttfts else 0.0

    @property
    def p99_ttft_ms(self) -> float:
        ttfts = sorted(r.ttft_ms for r in self.results)
        idx = max(0, int(len(ttfts) * 0.99) - 1)
        return ttfts[idx] if ttfts else 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report with metadata."""

    backend: str = "nanoserve"
    model: str = "toy"
    device: str = "cpu"
    commit_hash: str = ""
    timestamp: str = ""
    sweeps: list[SweepResult] = field(default_factory=list)


async def run_single_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int,
    request_id: str,
) -> RequestResult:
    """Send a single request and measure timing."""
    start = time.perf_counter()

    res = await client.post(
        "/v1/completions",
        json={
            "model": "toy",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=30.0,
    )
    end = time.perf_counter()

    data = res.json()
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    e2e_ms = (end - start) * 1000
    # For non-streaming, TTFT ≈ e2e (we can't distinguish in non-streaming mode)
    ttft_ms = e2e_ms
    tpot_ms = e2e_ms / max(1, completion_tokens)

    return RequestResult(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_ms,
        tpot_ms=tpot_ms,
    )


async def run_sweep(
    base_url: str,
    concurrency: int,
    num_requests: int,
    prompt: str = "The meaning of life is",
    max_tokens: int = 32,
) -> SweepResult:
    """Run a fixed-concurrency sweep."""
    async with httpx.AsyncClient(base_url=base_url) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request(idx: int) -> RequestResult:
            async with semaphore:
                return await run_single_request(client, prompt, max_tokens, f"bench-{idx}")

        start = time.perf_counter()
        tasks = [limited_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    total_tokens = sum(r.completion_tokens for r in results)
    throughput = total_tokens / total_time if total_time > 0 else 0.0

    return SweepResult(
        concurrency=concurrency,
        num_requests=num_requests,
        total_time_s=total_time,
        throughput_tok_s=throughput,
        results=list(results),
    )


def get_commit_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


async def run_benchmark(
    base_url: str = "http://localhost:8000",
    concurrency_levels: list[int] | None = None,
    num_requests: int = 50,
    output_path: str = "results/benchmark.json",
) -> BenchmarkReport:
    """Run the full benchmark suite."""
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 16, 32]

    report = BenchmarkReport(
        commit_hash=get_commit_hash(),
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    for concurrency in concurrency_levels:
        print(f"[BENCH] Concurrency={concurrency}, Requests={num_requests}...")
        sweep = await run_sweep(base_url, concurrency, num_requests)
        report.sweeps.append(sweep)
        print(
            f"  Throughput: {sweep.throughput_tok_s:.1f} tok/s, "
            f"p50 TTFT: {sweep.p50_ttft_ms:.1f} ms, "
            f"p99 TTFT: {sweep.p99_ttft_ms:.1f} ms"
        )

    # Save results
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    report_dict = {
        "backend": report.backend,
        "model": report.model,
        "device": report.device,
        "commit_hash": report.commit_hash,
        "timestamp": report.timestamp,
        "sweeps": [
            {
                "concurrency": s.concurrency,
                "num_requests": s.num_requests,
                "total_time_s": round(s.total_time_s, 3),
                "throughput_tok_s": round(s.throughput_tok_s, 1),
                "p50_ttft_ms": round(s.p50_ttft_ms, 2),
                "p99_ttft_ms": round(s.p99_ttft_ms, 2),
            }
            for s in report.sweeps
        ],
    }

    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"[BENCH] Results saved to {out_path}")

    return report


if __name__ == "__main__":
    asyncio.run(run_benchmark())
