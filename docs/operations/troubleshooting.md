# nanoserve Operator Troubleshooting & Diagnostics Runbook

This guide covers systematic diagnosis, root cause analysis, and remediation for production operational incidents when running `nanoserve`.

---

## 1. Out-of-Memory (OOM) Errors

### Symptoms:
- Error message: `RuntimeError: Block allocator OOM: need X more blocks, 0 free out of Y total`.
- Engine fails to admit new requests or preemption occurs repeatedly.

### Root Cause Analysis:
1. **Under-provisioned GPU Blocks**: `CacheConfig.num_gpu_blocks` is configured smaller than the peak concurrent active token working set ($B \times L_{avg}$).
2. **Prefix Cache Exhaustion**: All unreferenced blocks in LRU pool are pinned or referenced by running requests.

### Remediation Steps:
1. Increase `num_gpu_blocks` in `CacheConfig` or allocate dynamic memory using `available_bytes // kv_bytes_per_block`.
2. Configure preemption mode to `"swap"` and provision sufficient `num_cpu_blocks` in host RAM.
3. Lower `max_num_seqs` or `max_num_batched_tokens` in `SchedulerConfig` to constrain peak concurrency.

---

## 2. Preemption Thrashing

### Symptoms:
- High preemption counts in Prometheus metrics (`nanoserve_num_preemptions_total`).
- Degraded Time-Per-Output-Token (TPOT) and request latency variance.

### Root Cause Analysis:
- Engine admitted too many high-context sequences simultaneously, exceeding physical block budget during continuous decoding.

### Remediation Steps:
1. Switch to `"swap"` preemption to avoid discarded recompute FLOPs:
   ```python
   scheduler_config = SchedulerConfig(preemption_mode="swap")
   ```
2. Enable chunked prefill (`max_num_batched_tokens`) to smooth memory demand spikes.
3. Increase block allocator safety margin (`max_paddings`).

---

## 3. Triton Paged Attention Compilation Issues

### Symptoms:
- Server startup delay or runtime error when initializing CUDA kernel on unsupported GPU architectures.

### Root Cause Analysis:
- Triton requires Compute Capability $\ge 7.0$ (Volta, Turing, Ampere, Ada, Hopper). On unsupported hardware or CPU execution, Triton JIT will fail.

### Remediation Steps:
- `nanoserve` includes automated **Dual-Path Dynamic Dispatch**. If CUDA or Triton is unavailable, the engine automatically falls back to optimized PyTorch CPU reference attention:
  ```python
  from nanoserve.kernels.paged_attention import paged_attention_decode
  # Automatically routes to Triton on CUDA and PyTorch on CPU
  ```

---

## 4. Latency Spikes (TTFT or TPOT Degraded)

### Symptoms:
- P99 Time-To-First-Token (TTFT) $> 100$ ms.
- P99 Time-Per-Output-Token (TPOT) spikes during long prompt prefill.

### Root Cause Analysis:
- Prefill batch is monopolizing GPU compute cores and blocking single-token decode sequences.

### Remediation Steps:
1. Enable **SARATHI-style Chunked Prefill**:
   ```python
   scheduler_config = SchedulerConfig(max_num_batched_tokens=512)
   ```
2. Enable **Content-Addressed Prefix Caching** to achieve 0-FLOP prefill for shared prompt templates.
