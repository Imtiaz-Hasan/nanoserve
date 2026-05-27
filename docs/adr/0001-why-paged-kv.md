# ADR 0001: Why Paged KV Cache

## Status
Proposed (Week 2)

## Context
LLM inference requires storing key-value (KV) activations for all previously generated tokens. Naive implementations allocate a contiguous tensor per sequence sized to `max_sequence_length`, which wastes memory on sequences that terminate early and limits the number of concurrent sequences.

## Decision
Implement paged KV cache following the PagedAttention design (Kwon et al., SOSP 2023):
- Pre-allocate a pool of fixed-size blocks (e.g., 16 tokens each)
- Allocate blocks to sequences on demand as they grow
- Free blocks immediately when sequences finish
- Enable copy-on-write for prefix sharing

## Consequences

### Benefits
- **Near-zero internal fragmentation**: blocks are small and reusable
- **Higher concurrency**: memory is shared efficiently across sequences
- **Prefix caching**: content-addressed blocks enable free prefix reuse

### Costs
- **Complexity**: block tables, refcounting, and COW add implementation surface
- **Attention kernel**: standard SDPA can't handle non-contiguous KV; requires either gather-then-SDPA (slow) or a custom paged attention kernel (Week 8)

## Quantified Impact
To be measured in Week 2. Expected: KV memory utilization from ~30-40% (contiguous) to >90% (paged).
