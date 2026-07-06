"""Async serving benchmark client measuring TTFT, TPOT, and throughput via HTTP streaming."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import Any

import httpx

from benchmarks.report_generator import BenchmarkResult, compute_percentiles, save_benchmark_report


async def send_streaming_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    max_tokens: int = 32,
) -> dict[str, Any]:
    """Send a single streaming completion request and measure TTFT, TPOT, and latency."""
    url = f"{base_url}/v1/completions"
    payload = {
        "model": "toy",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time: float | None = None
    output_tokens = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=60.0) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[len("data: ") :].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            output_tokens += 1
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                    except json.JSONDecodeError:
                        continue
    except Exception as err:
        return {
            "success": False,
            "error": str(err),
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "latency_ms": 0.0,
            "output_tokens": 0,
        }

    end_time = time.perf_counter()
    first_token_time = first_token_time or end_time

    ttft_ms = (first_token_time - start_time) * 1000.0
    latency_ms = (end_time - start_time) * 1000.0

    if output_tokens > 1:
        tpot_ms = ((end_time - first_token_time) * 1000.0) / (output_tokens - 1)
    else:
        tpot_ms = 0.0

    return {
        "success": True,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "latency_ms": latency_ms,
        "output_tokens": output_tokens,
    }


async def run_serving_benchmark(
    base_url: str = "http://localhost:8000",
    num_requests: int = 32,
    concurrency: int = 8,
    request_rate: float | None = None,
    prompt_len: int = 64,
    output_len: int = 32,
) -> BenchmarkResult:
    """Run concurrent or Poisson serving benchmark against nanoserve server."""
    prompt = "Artificial intelligence and deep learning systems " * (prompt_len // 6)
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker() -> dict[str, Any]:
        async with semaphore:
            return await send_streaming_request(
                client=client,
                base_url=base_url,
                prompt=prompt,
                max_tokens=output_len,
            )

    async with httpx.AsyncClient() as client:
        start_bench = time.perf_counter()
        tasks: list[asyncio.Task[dict[str, Any]]] = []

        if request_rate is not None and request_rate > 0:
            # Poisson arrival process
            for _ in range(num_requests):
                task = asyncio.create_task(_worker())
                tasks.append(task)
                interval = random.expovariate(request_rate)
                await asyncio.sleep(interval)
        else:
            # Burst concurrency
            for _ in range(num_requests):
                tasks.append(asyncio.create_task(_worker()))

        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start_bench

    valid_results = [r for r in results if r.get("success", False)]
    ttft_list = [r["ttft_ms"] for r in valid_results]
    tpot_list = [r["tpot_ms"] for r in valid_results]
    latency_list = [r["latency_ms"] for r in valid_results]
    total_out = sum(r["output_tokens"] for r in valid_results)
    total_in = len(valid_results) * prompt_len

    num_valid = len(valid_results)
    req_tput = num_valid / duration if duration > 0 else 0.0
    tok_tput = total_out / duration if duration > 0 else 0.0

    mode = f"poisson_{request_rate}rps" if request_rate else f"burst_c{concurrency}"

    return BenchmarkResult(
        benchmark_name=f"serving_{mode}_p{prompt_len}_o{output_len}_n{num_requests}",
        num_requests=num_valid,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        duration_s=duration,
        request_throughput_req_per_s=req_tput,
        token_throughput_tok_per_s=tok_tput,
        ttft_ms=compute_percentiles(ttft_list),
        tpot_ms=compute_percentiles(tpot_list),
        latency_ms=compute_percentiles(latency_list),
    )


def main() -> None:
    """CLI entrypoint for serving benchmark."""
    parser = argparse.ArgumentParser(description="nanoserve serving benchmark client")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=None)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="benchmark_output")
    args = parser.parse_args()

    result = asyncio.run(
        run_serving_benchmark(
            base_url=args.base_url,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            request_rate=args.request_rate,
            prompt_len=args.prompt_len,
            output_len=args.output_len,
        )
    )

    save_benchmark_report([result], args.output_dir)
    print(f"\nBenchmark completed in {result.duration_s:.2f}s:")
    print(f"  Token Throughput: {result.token_throughput_tok_per_s:.2f} tok/s")
    print(f"  TTFT (P50 / P99): {result.ttft_ms['p50']:.2f} ms / {result.ttft_ms['p99']:.2f} ms")
    print(f"  TPOT (P50 / P99): {result.tpot_ms['p50']:.2f} ms / {result.tpot_ms['p99']:.2f} ms")


if __name__ == "__main__":
    main()
