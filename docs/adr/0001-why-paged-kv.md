# ADR 0001: Why Paged KV Cache

## Status
Accepted (Week 2 — 2026-06-03)

## Context
In autoregressive transformer serving, Key-Value (KV) cache tensors consume the majority of GPU VRAM. In traditional serving setups, memory is allocated contiguously for each sequence sized to `max_sequence_len` (e.g. 2048 or 4096 tokens). Because actual output lengths vary unpredictably (ShareGPT distribution has high variance), contiguous allocation suffers from severe internal and external memory fragmentation (often 60–70% of memory remains allocated but unused).

## Decision
Implement a virtual-memory-inspired **Paged KV Cache** (following Kwon et al., SOSP 2023):
1. Pre-allocate a global pool of fixed-size physical blocks (`num_blocks × block_size × num_kv_heads × head_dim`).
2. Maintain per-sequence dynamic `BlockTable` mapping logical block indices $\lfloor \text{token\_pos} / \text{block\_size} \rfloor$ to physical block IDs.
3. Support reference-counted physical block sharing and Copy-On-Write (COW) during sequence branching / beam search.
4. Provide paged cache scatter (`reshape_and_cache`) and gather (`gather_paged_kv`) routines for execution.

## Memory Arithmetic & Fragmentation Reduction

### Contiguous Allocation (Static Pre-allocation)
For $N$ sequences with maximum context $L_{\max}$ and actual length $L_i$:
$$\text{Memory}_{\text{contiguous}} = N \times L_{\max} \times 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{sizeof}(\text{dtype})$$
Internal fragmentation ratio:
$$F_{\text{internal}} = 1 - \frac{\sum_{i=1}^N L_i}{N \times L_{\max}} \approx 60\text{--}75\%$$

### Paged Allocation
With block size $B = 16$:
$$\text{Memory}_{\text{paged}} = \sum_{i=1}^N \lceil L_i / B \rceil \times B \times 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{sizeof}(\text{dtype})$$
Waste is strictly bounded by at most $B - 1 = 15$ tokens per active sequence:
$$F_{\text{paged}} \le \frac{B - 1}{\bar{L}} < 4\% \quad (\text{for } \bar{L} \ge 400)$$

## Consequences

### Benefits
- **>90% Memory Utilization**: Unlocks up to 3–5× higher serving concurrency on identical GPU memory budgets.
- **Copy-On-Write Sharing**: Branching and multi-turn prefixes share physical memory with zero duplicate bytes until divergence.
- **Zero Memory Leaks**: Enforced by Hypothesis property tests guaranteeing 100% block conservation.

### Trade-offs
- **Gather Overhead in v1**: PyTorch SDPA requires gathering non-contiguous blocks into a contiguous tensor before attention. This serves as the correctness oracle until the custom Triton PagedAttention kernel is introduced in Week 8.
