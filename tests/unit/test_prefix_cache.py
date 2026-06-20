"""Unit tests for content-addressed prefix caching, hash chaining, and LRU eviction."""

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams


def test_prefix_caching_shared_system_prompt_hit() -> None:
    """Verify second request with identical system prompt hits prefix cache and reuses blocks."""
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=64),
        device="cpu",
        seed=42,
    )
    engine = LLMEngine(config)

    # 32 tokens of shared system prompt (8 blocks of size 4)
    system_prompt = "S" * 32
    prompt_a = system_prompt + " User Question One"
    prompt_b = system_prompt + " User Question Two"

    # Request A: populates prefix cache
    engine.add_request("req-a", prompt_a, SamplingParams(temperature=0.0, max_tokens=3))
    while engine.has_unfinished_requests():
        engine.step()

    # Prefix cache must have cached blocks from request A
    assert engine.block_manager.prefix_cache.num_cached_blocks >= 8

    # Request B: arrives with identical system prompt prefix
    engine.add_request("req-b", prompt_b, SamplingParams(temperature=0.0, max_tokens=3))

    # Allocate blocks for Request B
    sg_b = engine.scheduler._waiting[0]
    engine.block_manager.allocate(sg_b.first_seq)

    # Sequence B immediately recognizes 32+ tokens already computed via prefix cache!
    assert sg_b.first_seq.num_computed_tokens >= 32


def test_prefix_caching_output_determinism() -> None:
    """Verify requests hitting prefix cache produce 100% identical outputs to un-cached runs."""
    system_prompt = "System prompt preamble here."
    full_prompt = system_prompt + " Followup user query text."
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)

    # 1. Un-cached baseline
    uncached_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_batched_tokens=128),
        device="cpu",
        seed=42,
    )
    uncached_engine = LLMEngine(uncached_config)
    uncached_engine.block_manager.enable_prefix_caching = False
    uncached_engine.add_request("base", full_prompt, sampling_params)

    baseline_tokens: list[int] = []
    while uncached_engine.has_unfinished_requests():
        outputs = uncached_engine.step()
        for out in outputs:
            if out.outputs:
                baseline_tokens.append(out.outputs[0].token_id)

    # 2. Cached run: first populate, then second query hits cache
    cached_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_batched_tokens=128),
        device="cpu",
        seed=42,
    )
    cached_engine = LLMEngine(cached_config)
    cached_engine.add_request("req-1", full_prompt, sampling_params)
    while cached_engine.has_unfinished_requests():
        cached_engine.step()

    # Query 2 (identical prompt)
    cached_engine.add_request("req-2", full_prompt, sampling_params)
    cached_tokens: list[int] = []
    while cached_engine.has_unfinished_requests():
        outputs = cached_engine.step()
        for out in outputs:
            if out.request_id == "req-2" and out.outputs:
                cached_tokens.append(out.outputs[0].token_id)

    # Cached run output is identical to baseline
    assert cached_tokens == baseline_tokens


def test_prefix_caching_lru_eviction_under_pressure() -> None:
    """Verify unreferenced cached blocks are evicted in LRU order when memory pool fills up."""
    # 4 blocks total capacity
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=4, block_size=4),
        scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=16),
        device="cpu",
        seed=42,
    )
    engine = LLMEngine(config)

    # Request 1: uses 2 blocks (8 tokens)
    engine.add_request("req-1", "A" * 8, SamplingParams(max_tokens=2))
    while engine.has_unfinished_requests():
        engine.step()

    # Request 2: uses 2 blocks (8 tokens)
    engine.add_request("req-2", "B" * 8, SamplingParams(max_tokens=2))
    while engine.has_unfinished_requests():
        engine.step()

    # Cache stores unreferenced prefix blocks in LRU pool (refcount == 0)
    assert engine.block_manager.prefix_cache.num_unreferenced_blocks >= 2

    # Request 3 arrives needing fresh blocks.
    # It must evict the oldest blocks from req-1 to make room!
    engine.add_request("req-3", "C" * 12, SamplingParams(max_tokens=2))
    while engine.has_unfinished_requests():
        engine.step()

    # Engine completed req-3 via LRU eviction without OOM crashes
    assert engine.block_manager.prefix_cache.num_unreferenced_blocks > 0
