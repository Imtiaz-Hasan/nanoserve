# ADR-0003: Chunked Prefill vs. Disaggregated Prefill-Decode Serving

## Status
**Accepted** (2026-07-01)

## Context
In LLM inference serving, requests exhibit a fundamental duality:
1. **Prefill Phase**: Compute-bound operation processing all prompt tokens simultaneously with high arithmetic intensity.
2. **Decode Phase**: Memory-bandwidth-bound operation generating one token at a time with low arithmetic intensity.

When a long prompt (e.g., 4,096 tokens) arrives while multiple sequences are in the decode phase, an un-chunked engine processes the entire prompt in a single forward pass lasting 100–300 ms. During this window, all running decode sequences are completely starved of compute, causing severe Time-Per-Output-Token (TPOT) tail latency spikes ($P_{99}$ degradation).

Two architectural paradigms exist to eliminate prefill-induced decode starvation:
- **In-Situ Chunked Prefill** (SARATHI-Serve, Agrawal et al., 2024; vLLM)
- **Disaggregated Prefill-Decode Architecture** (Splitwise, Patel et al., 2024; Mooncake, Qin et al., 2024; DistServe, Zhong et al., 2024)

---

## Architectural Comparison & Trade-Offs

### 1. In-Situ Chunked Prefill (Adopted by `nanoserve`)
- **Mechanism**: Splits long prompts into bounded chunks ($C \le \text{max\_num\_batched\_tokens}$, e.g. 512 tokens).
- Co-schedules prefill chunks alongside 1 token from each running decode sequence in the exact same model forward pass.
- **Hardware Synergy**: Combines the compute-heavy FLOPs of the prefill chunk with the memory-heavy reads of the decode sequences, achieving near-optimal GPU utilization (saturating both Tensor Cores and HBM bandwidth).
- **Network Overhead**: Zero network traffic; everything executes locally on the serving instance.

### 2. Disaggregated Prefill-Decode Serving
- **Mechanism**: Physically separates instances into dedicated **Prefill Workers** (optimized for compute) and **Decode Workers** (optimized for KV capacity and memory bandwidth).
- Once prefill completes, the full physical KV cache tensor is serialized and transferred across the network to a decode worker via RDMA (RoCE v2 or InfiniBand).
- **Network Requirement**: An 8B model with 4,096 context tokens produces 512 MiB of KV cache. Over a standard 25 Gbps Ethernet network, transferring this takes $\sim 170 \text{ ms}$ (worse than in-situ execution). It requires expensive 400 Gbps InfiniBand/RoCE infrastructure to achieve $< 15 \text{ ms}$ KV transfer.

---

## Decision Matrix

| Dimension | In-Situ Chunked Prefill | Disaggregated Prefill-Decode |
| :--- | :--- | :--- |
| **Cluster Topology** | Homogeneous standalone nodes | Heterogeneous pools + RDMA fabric |
| **Network Dependency** | None (Zero network transfer) | 400Gbps RoCE / InfiniBand mandatory |
| **Tail TPOT ($P_{99}$)** | $\sim 85–90\%$ reduction vs baseline | $\sim 95\%$ reduction vs baseline |
| **Implementation Complexity** | Low-to-Medium (Scheduler + Causal Mask) | High (Distributed state, KV RDMA RPCs) |
| **Cost Efficiency** | Optimal for 1–64 GPU deployments | Cost-effective only at datacenter scale ($> 100$ GPUs) |

---

## Decision

`nanoserve` implements **In-Situ Chunked Prefill with Mixed-Batch Scheduling**:
1. Unified token budgeting with decode priority guarantees zero starvation for running decodes.
2. Unified attention kernel with causal chunk masking enables seamless co-execution of prefill chunks and decode tokens in a single forward pass.
3. No external distributed network dependencies or RDMA hardware required.
