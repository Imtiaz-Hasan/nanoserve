"""Unit tests for chunked prefill (SARATHI mixed batches and zero decode starvation)."""

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams


def test_multi_iteration_chunked_prefill_progress() -> None:
    """Verify a prompt exceeding the token budget takes multiple iterations to prefill."""
    # Chunk budget = 16 tokens
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=16),
        device="cpu",
        seed=42,
    )
    engine = LLMEngine(config)

    # 64-byte prompt -> exactly 64 tokens -> requires 4 iterations of 16 tokens
    prompt_64 = "X" * 64
    engine.add_request("req-chunked", prompt_64, SamplingParams(temperature=0.0, max_tokens=2))

    # Step 1: processes tokens 0..16
    outputs1 = engine.step()
    assert len(outputs1) == 0  # prefill incomplete, no token emitted yet

    # Step 2: processes tokens 16..32
    outputs2 = engine.step()
    assert len(outputs2) == 0

    # Step 3: processes tokens 32..48
    outputs3 = engine.step()
    assert len(outputs3) == 0

    # Step 4: processes tokens 48..64 -> prefill complete, emits token 1!
    outputs4 = engine.step()
    assert len(outputs4) == 1
    assert outputs4[0].request_id == "req-chunked"
    assert not outputs4[0].finished

    # Step 5: decode step -> emits token 2 -> finished!
    outputs5 = engine.step()
    assert len(outputs5) == 1
    assert outputs5[0].finished


def test_no_decode_starvation_with_concurrent_long_prefill() -> None:
    """Verify active decode requests generate tokens on every step during long prefill."""
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=16),
        device="cpu",
        seed=42,
    )
    engine = LLMEngine(config)

    # Start 2 active decode requests
    engine.add_request("dec-1", "Prompt A", SamplingParams(temperature=0.0, max_tokens=8))
    engine.add_request("dec-2", "Prompt B", SamplingParams(temperature=0.0, max_tokens=8))

    # Prefill steps for decode requests
    engine.step()

    # Add long 48-token request (budget=16, minus 2 decodes = 14 tokens/step)
    engine.add_request("long-prefill", "Z" * 48, SamplingParams(temperature=0.0, max_tokens=2))

    # Run consecutive steps: dec-1 and dec-2 must produce tokens on EVERY step without stalling
    for _step in range(3):
        step_outputs = engine.step()
        output_req_ids = {out.request_id for out in step_outputs}
        assert "dec-1" in output_req_ids, "Decode sequence dec-1 was starved during chunked prefill"
        assert "dec-2" in output_req_ids, "Decode sequence dec-2 was starved during chunked prefill"


def test_chunked_prefill_output_parity() -> None:
    """Verify chunked prefill produces identical tokens to monolithic unchunked prefill."""
    prompt = "This is a prompt that will be tested for exact output equivalence across chunk sizes."
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # 1. Monolithic run (budget = 2048)
    mono_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_batched_tokens=2048),
        device="cpu",
        seed=42,
    )
    mono_engine = LLMEngine(mono_config)
    mono_engine.add_request("mono", prompt, sampling_params)

    mono_tokens: list[int] = []
    while mono_engine.has_unfinished_requests():
        outputs = mono_engine.step()
        for out in outputs:
            if out.outputs:
                mono_tokens.append(out.outputs[0].token_id)

    # 2. Chunked run (budget = 8 tokens per chunk)
    chunked_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_batched_tokens=8),
        device="cpu",
        seed=42,
    )
    chunked_engine = LLMEngine(chunked_config)
    chunked_engine.add_request("chunked", prompt, sampling_params)

    chunked_tokens: list[int] = []
    while chunked_engine.has_unfinished_requests():
        outputs = chunked_engine.step()
        for out in outputs:
            if out.outputs:
                chunked_tokens.append(out.outputs[0].token_id)

    # Outputs match byte-for-byte
    assert chunked_tokens == mono_tokens
