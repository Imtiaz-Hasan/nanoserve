# ADR-0002: Preemption Strategy — Recompute vs. Host Memory Swap

## Status
**Accepted** (2026-06-24)

## Context
Under bursty traffic or concurrent long-context requests, GPU physical KV cache memory can become completely exhausted during iterative decoding. Without an eviction mechanism, new token allocation will crash the engine with an unrecoverable Out-Of-Memory (OOM) error.

To guarantee zero OOM crashes under arbitrary load, the scheduler requires a preemption policy. Two architectural strategies exist:
1. **Recompute Preemption**: Drop the sequence's GPU physical blocks entirely and move the request back to the `waiting` queue. When GPU capacity frees up, re-run prompt prefill across `all_token_ids` (original prompt plus all tokens generated prior to eviction).
2. **Swap Preemption**: Allocate blocks from a secondary CPU host memory pool (`num_cpu_blocks`) and transfer the physical KV cache tensors from GPU to host memory via asynchronous PCIe DMA transfers. When GPU capacity frees up, swap the blocks back to device memory.

---

## Quantitative Trade-Off Analysis

Let $N$ be the number of context tokens accumulated by the evicted sequence.

### 1. KV Cache Footprint
For a Llama-3-8B model (32 layers, 8 GQA KV heads, $d_k = 128$, FP16):
$$\text{KV bytes per token} = 2 \times L \times H_{kv} \times d_k \times 2 = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ bytes} = 128 \text{ KiB/token}$$

For an $N = 4,096$ token sequence, the total KV cache footprint is:
$$\text{Memory} = 4096 \times 128 \text{ KiB} = 512 \text{ MiB}$$

### 2. Swap Latency (PCIe Gen4 x16)
PCIe Gen4 x16 provides effective unidirectional bandwidth of $B_{\text{pcie}} \approx 31.5 \text{ GB/s}$.
$$T_{\text{swap\_out}} + T_{\text{swap\_in}} = 2 \times \frac{512 \times 10^6 \text{ bytes}}{31.5 \times 10^9 \text{ bytes/sec}} \approx 32.5 \text{ ms}$$

Per-token transfer overhead: $\approx 7.9 \ \mu\text{s/token}$.

### 3. Recompute Latency (FLOP Bound on Modern GPU)
Prefill compute on an 8B model requires $\approx 2 \times 8 \times 10^9 = 16 \text{ GFLOPs/token}$.
On an NVIDIA H100 GPU (effective dense FP16 throughput $\approx 1,000 \text{ TFLOPs} = 10^{15} \text{ FLOP/s}$):
$$T_{\text{recompute}}(N) = \frac{16 \times 10^9 \times N}{10^{15}} = N \times 16 \ \mu\text{s/token}$$

For $N = 4,096$ tokens:
$$T_{\text{recompute}} = 4096 \times 16 \ \mu\text{s} \approx 65.5 \text{ ms}$$

---

## Decision

| Metric | Recompute Mode | Swap Mode |
| :--- | :--- | :--- |
| **GPU Compute Overhead** | Burns GPU tensor cores ($16 \ \mu\text{s/token}$) | Zero tensor core compute overhead |
| **PCIe Bandwidth** | Zero PCIe traffic | Saturated during transfer ($32.5 \text{ ms} / 512 \text{ MiB}$) |
| **Host Memory Footprint** | Zero CPU RAM required | Requires pre-allocated host pinned memory |
| **Optimal Use Case** | Short contexts ($N < 1024$) or compute-idle servers | Long contexts ($N > 2048$) or heavily loaded clusters |

`nanoserve` implements **both** strategies behind a configurable scheduler toggle (`SchedulerConfig.preemption_mode`):
- `recompute` (default for CPU/single-node development): lightweight, requires zero host cache allocation.
- `swap`: high-throughput mode utilizing `BlockManager` CPU pools and `swap_blocks` asynchronous transfers.

## Consequences
- **Zero OOM guarantee**: No incoming request volume or context length can crash the engine.
- **Priority inversion prevention**: Swapped/recomputed requests maintain higher priority than newly arriving waiting requests to minimize time-to-first-token degradation.
