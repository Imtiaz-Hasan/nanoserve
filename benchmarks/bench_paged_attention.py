"""Benchmark: Paged Attention decode kernel latency and memory bandwidth."""

from __future__ import annotations

import argparse
import math
import time

import torch

from nanoserve.kernels.paged_attention import paged_attention_decode


def benchmark_paged_attention(
    batch_size: int,
    seq_len: int,
    num_heads: int = 32,
    head_dim: int = 128,
    block_size: int = 16,
    num_warmup: int = 5,
    num_iters: int = 20,
    device: str = "cpu",
) -> dict[str, float]:
    """Benchmark paged attention decode latency and bandwidth for given dimensions."""
    dev = torch.device(device)
    num_blocks = math.ceil(seq_len / block_size) * batch_size + 16

    k_cache = torch.randn(num_blocks, num_heads, block_size, head_dim, device=dev)
    v_cache = torch.randn(num_blocks, num_heads, block_size, head_dim, device=dev)
    q = torch.randn(batch_size, num_heads, 1, head_dim, device=dev)

    blocks_per_seq = math.ceil(seq_len / block_size)
    block_tables = [
        list(range(i * blocks_per_seq, (i + 1) * blocks_per_seq)) for i in range(batch_size)
    ]
    seq_lens = [seq_len] * batch_size

    # Warmup
    for _ in range(num_warmup):
        _ = paged_attention_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )

    if dev.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(num_iters):
        _ = paged_attention_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
        )

    if dev.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / num_iters) * 1000.0

    # Total KV bytes loaded: 2 * batch_size * seq_len * num_heads * head_dim * sizeof(float32)
    bytes_loaded = 2 * batch_size * seq_len * num_heads * head_dim * 4
    bandwidth_gbps = (bytes_loaded / (avg_latency_ms / 1000.0)) / (1024**3)

    return {
        "batch_size": float(batch_size),
        "seq_len": float(seq_len),
        "avg_latency_ms": avg_latency_ms,
        "bandwidth_gbps": bandwidth_gbps,
    }


def main() -> None:
    """Run sweep over batch sizes and context lengths."""
    parser = argparse.ArgumentParser(description="Paged Attention Decode Benchmark")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Running Paged Attention Benchmark on device: {device}\n")
    print(f"{'Batch':<8} {'SeqLen':<10} {'Latency (ms)':<15} {'Bandwidth (GB/s)':<18}")
    print("-" * 55)

    for batch_size in [1, 8, 32]:
        for seq_len in [128, 512, 2048]:
            res = benchmark_paged_attention(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
            )
            print(
                f"{int(res['batch_size']):<8} {int(res['seq_len']):<10} "
                f"{res['avg_latency_ms']:<15.3f} {res['bandwidth_gbps']:<18.2f}"
            )


if __name__ == "__main__":
    main()
