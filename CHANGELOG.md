# Changelog

All notable changes to this project will be documented in this file.

## [0.5.0] — 2026-06-24

### Added
- **Preemption Subsystem**: Dual eviction modes (Recompute & CPU memory Swap) protecting against GPU OOM errors
- **Dual-Pool Memory Allocation**: `BlockManager` support for CPU host block pools alongside GPU block pools
- **Asynchronous KV Transfer Kernel**: `swap_blocks` for physical KV cache memory transfers between host and device
- **LIFO Victim Selection**: Prioritized eviction with resume precedence (`swapped` > `running` > `waiting`)
- **ADR-0002**: Mathematical trade-off analysis between PCIe bandwidth and recompute FLOPs
- **Preemption Tests**: Verification under extreme memory pressure (4 blocks / 8 requests) with 100% output determinism

## [0.4.0] — 2026-06-17

### Added
- **Continuous Batching**: Iteration-level scheduler managing waiting, running, and swapped queues (Orca-style)
- **Multi-Sequence Batched Decode**: Single parallel GPU/CPU forward pass for all active decode sequences
- **Dynamic Admission**: New requests admitted to active batch immediately after prefill without waiting for long sequences to finish
- **Zero Token Padding**: Full FLOP efficiency across variable length requests with zero pad tokens
- **Continuous Batching Tests**: Verifies asynchronous completion order and dynamic mid-flight request admission

## [0.3.0] — 2026-06-10

### Added
- **Full Sampling Suite**: Temperature scaling, Top-K truncation, Top-P (nucleus) filtering, and Min-P relative thresholding
- **Penalties Module**: Multiplicative repetition penalty and additive frequency/presence penalties (`nanoserve.sampling.penalties`)
- **Seeded Generator**: Isolated per-request `torch.Generator` for 100% deterministic reproducibility
- **Statistical Correctness Gate**: Automated tests for nucleus mass confinement, low-T argmax convergence, and high-T uniformity
- **Robust Stop Strings**: Multi-byte UTF-8 emoji detection and overlapping prefix stop criteria

## [0.2.0] — 2026-06-03

### Added
- **Paged KV Cache**: Full `BlockManager` with physical block tracking, refcounting, and Copy-On-Write (COW)
- **Dynamic BlockTable**: Logical-to-physical block mapping with sequence branching support
- **Paged Kernel Operations**: `reshape_and_cache`, `gather_paged_kv`, and `copy_block_data`
- **Property Fuzz Testing**: Hypothesis test suite verifying 5 memory conservation invariants across 10k random operations
- **ADR-0001 Accepted**: Comprehensive memory fragmentation analysis and architecture decision record

## [0.1.0] — 2026-05-27

### Added
- Initial vertical slice: weight loading, Llama forward pass, greedy generation
- OpenAI-compatible API (`/v1/completions`, `/v1/chat/completions`)
- Naive contiguous KV cache baseline
- CPU-testable toy model (2-layer, 4-head, 64-dim)
- Prometheus metrics endpoint (`/metrics`)
- Structured logging with `structlog`
- CI pipeline: ruff, mypy --strict, pytest
- Dockerfile
