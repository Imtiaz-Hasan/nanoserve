# ADR-0004: Speculative Decoding Proposer Strategy

- **Status**: Accepted
- **Date**: 2026-08-12
- **Author**: Imtiaz Hasan
- **Deciders**: Systems Engineering Team

---

## 1. Context & Problem Statement

Autoregressive language model generation is memory-bandwidth bound: generating a single token requires reading all model weights from High Bandwidth Memory (HBM) into compute cores. Speculative decoding allows generating multiple tokens per target model memory read by having a fast *proposer* propose $K$ candidate tokens, which the *target* model verifies in parallel within a single forward pass.

The primary architectural question is: **Which candidate proposal strategy should `nanoserve` adopt as its core speculative subsystem?**

---

## 2. Considered Alternatives

### Option 1: Separate Lightweight Draft Language Model
- **Mechanism**: Run a smaller model (e.g., Llama-1B draft for Llama-8B target) for $K$ autoregressive steps.
- **Pros**: Generates high-quality candidate tokens across diverse open-ended tasks.
- **Cons**:
  - Requires loading and managing two distinct sets of model weights in GPU VRAM.
  - Draft model forward pass introduces compute and latency overhead.
  - Requires maintaining synchronized tokenizers between draft and target models.

### Option 2: N-Gram (Prompt-Lookup) Speculation
- **Mechanism**: Scan the input prompt and recent token generation history for recurring $N$-grams (e.g., $N=3$). If an exact match is found earlier in the sequence, propose the subsequent $K$ tokens as candidates.
- **Pros**:
  - **Zero FLOPs**: Pure string/token array slicing in CPU/host memory.
  - **Zero GPU Memory Overhead**: No secondary model weights or extra KV caches required.
  - High acceptance rate ($\alpha \approx 0.65 - 0.85$) on document summarization, retrieval-augmented generation (RAG), few-shot prompting, and code generation/editing.
  - Lossless guarantee when verified by target model.
- **Cons**:
  - Low acceptance rate on high-entropy creative writing where tokens rarely repeat from the prompt.

### Option 3: Medusa / Multi-Head Speculation
- **Mechanism**: Train auxiliary prediction heads atop the target model's final hidden states to predict future tokens.
- **Pros**: Single model instance, no separate draft model.
- **Cons**:
  - Requires custom training/fine-tuning for every supported model checkpoint.
  - Adds parameter overhead and prevents drop-in serving of standard open-weight checkpoints.

---

## 3. Decision Outcome

**Chosen Solution**: **Hybrid Strategy with N-Gram Prompt-Lookup Default**.

We adopt **N-Gram Prompt-Lookup Speculation** as the primary default proposer in `nanoserve` and support **Draft Model Proposers** as an optional pluggable backend.

### Rationale:
1. **Zero Resource Overhead**: N-gram prompt lookup requires 0 extra parameters and 0 extra GPU memory, making speculative decoding universally accessible on single-GPU and resource-constrained environments.
2. **Deterministic Parity**: Verification with rejection sampling guarantees 100% token-for-token mathematical equivalence with standard autoregressive generation.
3. **High Empirical Speedup**: In RAG and code editing workloads, prompt-lookup achieves $2.2\times - 2.8\times$ decoding speedup with zero training.

---

## 4. Consequences & Trade-offs

### Positive:
- Instant speedup for document analysis, coding, and RAG without downloading additional draft models.
- Clean architectural abstraction via `SpeculativeProposer` protocol supporting future proposers (Eagle, Medusa, Draft LM).
- Automatic KV cache rollback ensures memory safety on candidate rejection.

### Negative / Mitigations:
- When prompt lookup produces 0 candidate matches, the engine falls back to standard single-token decode with negligible ($< 0.05$ ms) overhead.
