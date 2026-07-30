# Changelog

All notable changes to this project will be documented in this file.

## [1.4.0] — 2026-07-30

### Added
- **Disaggregated Prefill-Decode Architecture**: Specialized compute-bound `PrefillWorker` and memory-bandwidth-bound `DecodeWorker` pools
- **KV Cache Transfer Protocol**: High-efficiency serialization and IPC/network payload transport (`KVTransferPayload`)
- **Disaggregated Router**: Central coordinator dispatching prompt prefills, transferring computed KV states, and managing token stream emission
- **Disaggregated Test Suite**: Binary transfer roundtrips, worker lifecycle handoffs, and end-to-end routing parity

## [1.3.0] — 2026-07-25

### Added
- **Tensor Parallelism (Megatron-LM Style)**: Multi-worker model execution sharding query, key, value, gate, and up projections
- **`ColumnParallelLinear`**: Zero-communication output dimension feature sharding ($D_{out} / TP$)
- **`RowParallelLinear`**: Input dimension sharding ($D_{in} / TP$) with collective All-Reduce aggregation
- **`VocabParallelEmbedding`**: Distributed token lookup across partitioned vocabulary shards
- **`DistributedCommunicator`**: Unified PyTorch distributed backend and in-process CPU mock communicator for deterministic multi-worker testing
- **Tensor Parallel Test Suite**: Exact mathematical parity across Column, Row, Vocab, and 2-layer SwiGLU MLP blocks

## [1.2.0] — 2026-07-20

### Added
- **Multi-LoRA Dynamic Adapter Hot-Swapping**: Zero-restart runtime loading, caching, and eviction of LoRA adapters on frozen base models
- **Heterogeneous Batched Serving**: `LoraLinear` computes per-token adapter deltas allowing different requests in the same batch to run different LoRA adapters concurrently
- **LoRA Manager**: Central lifecycle orchestrator managing adapter configurations, module injection, and metadata queries
- **Sequence Routing**: Per-request `lora_name` tracking through `Sequence` and `SequenceGroup`
- **LoRA Unit Test Suite**: Forward-pass parity, dynamic registration lifecycle, and heterogeneous batch isolation tests

## [1.1.0] — 2026-07-15

### Added
- **INT8/FP8 Quantized KV Cache**: Block-level symmetric quantization halving memory footprint and doubling concurrent serving capacity
- **Quantized Scatter/Gather**: High-fidelity dequantization pipeline with $>0.99$ cosine fidelity and $<0.05$ MAE attention parity
- **Constrained Guided Decoding**: Step-wise `RegexLogitProcessor` enforcing strict regular expressions
- **JSON Schema Logit Biasing**: `JsonSchemaLogitProcessor` enforcing structural JSON syntax and schema compliance
- **Quantization & Guided Tests**: Comprehensive test suite covering quantization error bounds and grammar masking

## [1.0.0] — 2026-07-10

### Added
- **Production GA Milestone (v1.0.0)**: Complete feature roadmap achieved
- **Architecture Whitepaper**: Complete systems and mathematical design document (`docs/architecture/whitepaper.md`)
- **ADR-0004**: Architectural Decision Record on Speculative Decoding Proposer Strategy (`docs/adr/0004-speculative-decoding-proposer-strategy.md`)
- **Operator Runbooks**: Diagnostic troubleshooting guide (`docs/operations/troubleshooting.md`) and deployment guide (`docs/operations/deployment.md`)
- **100% Test Coverage**: Comprehensive passing unit, integration, property, fuzz, and correctness golden tests

## [0.11.0] — 2026-07-06

### Added
- **Async Streaming Benchmark Client**: High-precision TTFT, TPOT, and latency percentile measurement over SSE
- **Traffic Generation Models**: Supports both burst concurrency and Poisson arrival processes ($\lambda$ req/s)
- **In-Process Throughput Benchmark**: Measures raw engine scheduling and generation speed without network stack overhead
- **Automated Report Generator**: Generates publication-ready Markdown tables, JSON metrics, and LaTeX tables
- **Benchmark Test Suite**: Unit and integration testing for statistical percentiles, synthetic workloads, and report export

## [0.10.0] — 2026-07-02

### Added
- **SafeTensors Weight Loader**: Zero-copy single-file and sharded checkpoint loading from local disk and HuggingFace Hub
- **Parameter Name Normalization**: Maps HuggingFace model architectures (LLaMA, Mistral, Qwen) directly to nanoserve tensors
- **HuggingFace AutoTokenizer Integration**: High-throughput subword tokenization with fallback to byte-level tokenizer
- **`LlamaForCausalLM.from_pretrained`**: End-to-end checkpoint instantiation from `config.json` and SafeTensors files
- **Golden Parity Testing**: Mathematical forward-pass numerical verification against PyTorch reference implementation

## [0.9.0] — 2026-06-28

### Added
- **N-Gram Prompt-Lookup Speculation**: Zero-overhead candidate generation scanning prompt history for recurrent patterns
- **Draft Model Speculative Proposer**: Multi-token candidate generation via lightweight draft language model
- **Parallel Speculative Verifier**: Single target forward pass verifying $K+1$ candidate positions with bonus token emission
- **KV Cache Rollback Mechanism**: Automatic cache slot and token rollback on candidate mismatch
- **Speculative Engine Orchestrator**: High-level engine managing proposal, verification, and acceptance rate ($\alpha$) metrics
- **Speculative Tests**: 100% token-for-token mathematical equivalence, speedup validation, and rollback safety

## [0.8.0] — 2026-06-24

### Added
- **Fused Triton Paged Attention Kernel**: Custom GPU decode attention kernel with single-pass online softmax
- **Dual-Path Dynamic Dispatch**: Automatically executes compiled Triton kernel on CUDA and PyTorch reference SDPA on CPU
- **Generalized GQA Expansion**: Dynamic runtime key/value head broadcasting across arbitrary query heads
- **Zero-Length Block Safety**: Defensive guarding against unallocated cache slots and ragged tail sequences
- **Kernel Correctness Suite**: Parameterized unit testing across batch sizes, head counts, and block sizes

## [0.7.0] — 2026-06-20

### Added
- **Content-Addressed Prefix Caching**: Deterministic hash-chaining of physical blocks based on prefix tokens
- **Unreferenced Block LRU Cache**: Retains reusable prefix blocks in memory across independent completed requests
- **Zero-Compute Prompt Admission**: New sequences matching cached system prompts skip prefill FLOPs
- **Dynamic Eviction Under Pressure**: Automatic eviction of oldest LRU blocks when memory pool is exhausted
- **Prefix Cache Testing**: Validated cache hits, output determinism, and high-pressure eviction invariants

## [0.6.0] — 2026-06-16

### Added
- **SARATHI-Style Chunked Prefill**: Splits long prompt evaluations into bounded token chunks ($C \le \text{max\_num\_batched\_tokens}$)
- **Mixed Prefill/Decode Batching**: Co-schedules prefill chunks alongside ongoing single-token decode requests in a single forward pass
- **Starvation-Free Scheduling**: Eliminates decode latency spikes caused by incoming long prompt prefills
- **Multi-Iteration Chunked State Tracking**: Sequences seamlessly track partial computed tokens across scheduler iterations
- **Chunked Prefill Golden Parity**: Validates identical token output generation between chunked and non-chunked inference

## [0.5.0] — 2026-06-12

### Added
- **Zero-OOM Preemption Engine**: Guaranteed crash-free operation under extreme KV cache memory exhaustion
- **Recompute Preemption Mode**: Suspends victim sequences, frees GPU KV blocks, and restarts from prompt on recovery
- **CPU Host Memory Swap Mode**: Asynchronously offloads physical KV cache blocks to host RAM and restores when GPU memory frees
- **Preemption Scheduler State Machine**: FCFS victim selection with automatic re-admission and queue promotion
- **Deterministic Preemption Tests**: Validates exact token-for-token reproducibility across multiple swap/recompute cycles

## [0.4.0] — 2026-06-08

### Added
- **Orca-Style Continuous Batching**: Iteration-level scheduling dynamic admission without sequence padding
- **Dynamic Multi-Sequence Decode**: Batches heterogeneous active sequences with variable prompt and generation lengths
- **Immediate Memory Reclamation**: Releases physical KV blocks to the pool the exact iteration a sequence completes
- **Multi-Stream Staggered Generation**: Verified correct token generation across asynchronous arrival and departure schedules

## [0.3.0] — 2026-06-04

### Added
- **Comprehensive Sampling Suite**: Temperature scaling, Top-K, Top-P (nucleus), and Min-P relative truncation
- **Penalties Pipeline**: Repetition penalty, additive frequency penalty, and presence penalty
- **Seeded RNG Reproducibility**: Deterministic sampling guarantees across CPU and GPU hardware
- **Advanced Stop Sequence Detection**: Multi-token string matching spanning across chunk and token boundaries

## [0.2.0] — 2026-05-31

### Added
- **Paged KV Cache Virtual Memory**: Non-contiguous block allocator eliminating external and internal memory fragmentation
- **Physical Block Tables**: Logical-to-physical address translation mapping sequences to physical HBM slots
- **Copy-On-Write (COW) Semantics**: Reference-counted block sharing for parallel sampling and beam search branching
- **Scatter/Gather Kernels**: Slot-mapping kernels for populating and gathering paged KV activations

## [0.1.0] — 2026-05-27

### Added
- Initial nanoserve vertical slice with LLaMA architecture in PyTorch
- Greedy deterministic decoding engine and basic block allocator
- FastAPI OpenAI-compatible `/v1/chat/completions` and `/v1/completions` server with SSE streaming
- Comprehensive CI test harness and strict type annotations
