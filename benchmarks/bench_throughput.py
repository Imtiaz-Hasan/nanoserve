"""In-process engine throughput benchmark for raw scheduling and generation speed."""

from __future__ import annotations

import argparse
import time

from benchmarks.report_generator import BenchmarkResult, compute_percentiles
from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams


def run_throughput_benchmark(
    num_requests: int = 32,
    prompt_len: int = 128,
    output_len: int = 32,
    block_size: int = 16,
    device: str = "cpu",
    dtype: str = "float32",
) -> BenchmarkResult:
    """Run in-process throughput benchmark on LLMEngine."""
    config = EngineConfig(
        model=ModelConfig(
            model_name_or_path="toy",
            dtype=dtype,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            hidden_size=256,
            intermediate_size=688,
            vocab_size=256,
        ),
        cache=CacheConfig(
            num_gpu_blocks=512,
            block_size=block_size,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=64,
            max_num_batched_tokens=2048,
        ),
        device=device,
        seed=42,
    )

    engine = LLMEngine(config)
    prompt_text = "test " * (prompt_len // 5)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=output_len)

    # Submit all requests
    req_start_times: dict[str, float] = {}
    req_first_token_times: dict[str, float] = {}
    req_end_times: dict[str, float] = {}
    req_output_counts: dict[str, int] = {}

    for i in range(num_requests):
        req_id = f"bench_req_{i}"
        req_start_times[req_id] = time.perf_counter()
        req_output_counts[req_id] = 0
        engine.add_request(req_id, prompt_text, sampling_params)

    bench_start = time.perf_counter()

    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        now = time.perf_counter()

        for out in step_outputs:
            req_id = out.request_id
            if out.outputs:
                req_output_counts[req_id] += 1
                if req_id not in req_first_token_times:
                    req_first_token_times[req_id] = now
                if out.finished:
                    req_end_times[req_id] = now

    bench_duration = time.perf_counter() - bench_start

    # Compute metrics
    ttft_list: list[float] = []
    tpot_list: list[float] = []
    latency_list: list[float] = []

    total_in_tokens = num_requests * len(engine.tokenizer.encode(prompt_text))
    total_out_tokens = sum(req_output_counts.values())

    for req_id, start_t in req_start_times.items():
        first_t = req_first_token_times.get(req_id, start_t)
        end_t = req_end_times.get(req_id, first_t)
        num_out = req_output_counts.get(req_id, 1)

        ttft_ms = (first_t - start_t) * 1000.0
        ttft_list.append(ttft_ms)

        e2e_ms = (end_t - start_t) * 1000.0
        latency_list.append(e2e_ms)

        if num_out > 1:
            tpot_ms = ((end_t - first_t) * 1000.0) / (num_out - 1)
            tpot_list.append(tpot_ms)
        else:
            tpot_list.append(0.0)

    req_tput = num_requests / bench_duration if bench_duration > 0 else 0.0
    tok_tput = total_out_tokens / bench_duration if bench_duration > 0 else 0.0

    return BenchmarkResult(
        benchmark_name=f"in_process_p{prompt_len}_o{output_len}_n{num_requests}",
        num_requests=num_requests,
        total_input_tokens=total_in_tokens,
        total_output_tokens=total_out_tokens,
        duration_s=bench_duration,
        request_throughput_req_per_s=req_tput,
        token_throughput_tok_per_s=tok_tput,
        ttft_ms=compute_percentiles(ttft_list),
        tpot_ms=compute_percentiles(tpot_list),
        latency_ms=compute_percentiles(latency_list),
    )


def main() -> None:
    """CLI entrypoint for in-process throughput benchmark."""
    parser = argparse.ArgumentParser(description="nanoserve throughput benchmark")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    res = run_throughput_benchmark(
        num_requests=args.num_requests,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        device=args.device,
    )

    print(f"\nThroughput: {res.token_throughput_tok_per_s:.2f} tok/s ({res.request_throughput_req_per_s:.2f} req/s)")
    print(f"TTFT P50: {res.ttft_ms['p50']:.2f} ms | P99: {res.ttft_ms['p99']:.2f} ms")
    print(f"TPOT P50: {res.tpot_ms['p50']:.2f} ms | P99: {res.tpot_ms['p99']:.2f} ms")


if __name__ == "__main__":
    main()
