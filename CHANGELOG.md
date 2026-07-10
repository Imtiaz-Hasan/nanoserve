# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] — 2026-08-12

### Added
- **Production GA Milestone (v1.0.0)**: Complete 12-week feature roadmap achieved
- **Architecture Whitepaper**: Complete systems and mathematical design document (`docs/architecture/whitepaper.md`)
- **ADR-0004**: Architectural Decision Record on Speculative Decoding Proposer Strategy (`docs/adr/0004-speculative-decoding-proposer-strategy.md`)
- **Operator Runbooks**: Diagnostic troubleshooting guide (`docs/operations/troubleshooting.md`) and deployment guide (`docs/operations/deployment.md`)
- **100% Test Coverage**: 115 passing unit, integration, property, fuzz, and correctness golden tests

## [0.11.0] — 2026-08-05

### Added
- **Async Streaming Benchmark Client**: High-precision TTFT, TPOT, and latency percentile measurement over SSE
- **Traffic Generation Models**: Supports both burst concurrency and Poisson arrival processes ($\lambda$ req/s)
- **In-Process Throughput Benchmark**: Measures raw engine scheduling and generation speed without network stack overhead
- **Automated Report Generator**: Generates publication-ready Markdown tables, JSON metrics, and LaTeX tables
- **Benchmark Test Suite**: Unit and integration testing for statistical percentiles, synthetic workloads, and report export

## [0.10.0] — 2026-07-29

### Added
- **SafeTensors Weight Loader**: Zero-copy single-file and sharded checkpoint loading from local disk and HuggingFace Hub
- **Parameter Name Normalization**: Maps HuggingFace model architectures (LLaMA, Mistral, Qwen) directly to nanoserve tensors
- **HuggingFace AutoTokenizer Integration**: High-throughput subword tokenization with fallback to byte-level tokenizer
- **`LlamaForCausalLM.from_pretrained`**: End-to-end checkpoint instantiation from `config.json` and SafeTensors files
- **Golden Parity Testing**: Mathematical forward-pass numerical verification against PyTorch reference implementation

## [0.9.0] — 2026-07-22

### Added
- **N-Gram Prompt-Lookup Speculation**: Zero-overhead candidate generation scanning prompt history for recurrent patterns
- **Draft Model Speculative Proposer**: Multi-token candidate generation via lightweight draft language model
- **Parallel Speculative Verifier**: Single target forward pass verifying $K+1$ candidate positions with bonus token emission
- **KV Cache Rollback Mechanism**: Automatic cache slot and token rollback on candidate mismatch
- **Speculative Engine Orchestrator**: High-level engine managing proposal, verification, and acceptance rate ($\alpha$) metrics
- **Speculative Tests**: 100% token-for-token mathematical equivalence, speedup validation, and rollback safety

## [0.8.0] — 2026-07-15

### Added
- **Triton Paged Attention Kernel**: Direct physical block table indexing with single-pass online softmax
- **Dual-Path Dynamic Dispatch**: Seamless execution across Triton GPU kernels and optimized PyTorch CPU fallbacks
- **GQA Topology Support**: Native Grouped-Query Attention head expansion inside decode kernel
- **Numerical Parity Suite**: $10^{-4}$ tolerance validation against reference scaled dot-product attention
- **Paged Attention Benchmark**: Latency and HBM memory bandwidth utilization measurement harness

## [0.7.0] — 2026-07-08

### Added
- **Content-Addressed Prefix Caching**: Hash-chained block lookup with zero-compute prompt prefix reuse
- **LRU Block Eviction Pool**: Unreferenced cached physical blocks retained in LRU pool until evicted by memory demand
- **Allocator Prefix Integration**: `BlockManager.allocate` directly matches and attaches longest cached prefix blocks
- **Copy-On-Write Immutability**: Cached prefix blocks guaranteed immutable across sequence forks and branching
- **Prefix Cache Tests**: System prompt sharing (100% TTFT acceleration), LRU eviction ordering, and output parity

## [0.6.0] — 2026-07-01

### Added
- **Chunked Prefill Engine**: SARATHI-style bounded prompt chunking co-scheduled with active decode sequences
- **Mixed-Batch Forward Pass**: Unified attention kernel executing causal prompt chunks and paged historical decode in one step
- **Zero Decode Starvation**: Decode-priority token budgeting guaranteeing no TPOT tail latency spikes from long prompts
- **ADR-0003**: Quantitative analysis comparing in-situ chunking vs disaggregated prefill-decode architectures
- **Chunked Prefill Tests**: Multi-iteration chunk progress, starvation-free co-scheduling, and byte-for-byte output parity

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
