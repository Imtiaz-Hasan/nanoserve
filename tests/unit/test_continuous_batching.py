"""Unit tests for continuous batching (iteration-level scheduling and dynamic admission)."""

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams


def create_test_engine(max_num_seqs: int = 8) -> LLMEngine:
    """Create an LLMEngine configured for multi-sequence continuous batching testing."""
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=16),
        scheduler=SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=2048),
        device="cpu",
        seed=42,
    )
    return LLMEngine(config)


def test_continuous_batching_staggered_completion() -> None:
    """Verify shorter requests finish earlier and unblock resources for others."""
    engine = create_test_engine(max_num_seqs=4)

    # Submit 3 requests: Req-Short (3 tokens), Req-Med (6 tokens), Req-Long (9 tokens)
    engine.add_request("req-short", "Short prompt", SamplingParams(temperature=0.0, max_tokens=3))
    engine.add_request(
        "req-med", "Medium prompt text", SamplingParams(temperature=0.0, max_tokens=6)
    )
    engine.add_request(
        "req-long", "Long prompt text here", SamplingParams(temperature=0.0, max_tokens=9)
    )

    completion_iterations: dict[str, int] = {}
    iteration = 0

    while engine.has_unfinished_requests() and iteration < 30:
        iteration += 1
        outputs = engine.step()
        for out in outputs:
            if out.finished and out.request_id not in completion_iterations:
                completion_iterations[out.request_id] = iteration

    assert "req-short" in completion_iterations
    assert "req-med" in completion_iterations
    assert "req-long" in completion_iterations

    # Assert shorter requests completed strictly before longer ones
    assert completion_iterations["req-short"] < completion_iterations["req-med"]
    assert completion_iterations["req-med"] < completion_iterations["req-long"]


def test_dynamic_admission_mid_flight() -> None:
    """Verify new requests submitted mid-flight join the running batch on the next iteration."""
    engine = create_test_engine(max_num_seqs=4)

    # Submit Request 1
    engine.add_request("req-1", "Initial request", SamplingParams(temperature=0.0, max_tokens=8))

    # Run 3 steps (1 prefill + 2 decodes)
    for _ in range(3):
        engine.step()

    # Now add Request 2 mid-flight
    engine.add_request(
        "req-2", "Second request arriving late", SamplingParams(temperature=0.0, max_tokens=4)
    )

    # Next step: should process prefill for Request 2
    step_outputs = engine.step()
    req_ids_in_step = {out.request_id for out in step_outputs}
    assert "req-2" in req_ids_in_step

    # Following step: both req-1 and req-2 run concurrently in decode batch!
    step_outputs = engine.step()
    req_ids_in_step = {out.request_id for out in step_outputs}
    assert "req-1" in req_ids_in_step
    assert "req-2" in req_ids_in_step


def test_memory_freed_immediately_on_finish() -> None:
    """Verify physical blocks of completed requests are freed immediately on finish."""
    engine = create_test_engine(max_num_seqs=4)
    initial_free_blocks = engine.block_manager.num_free_blocks

    engine.add_request("req-1", "Prompt", SamplingParams(temperature=0.0, max_tokens=2))

    # Prefill step (generates token 1)
    outputs1 = engine.step()
    assert len(outputs1) == 1
    assert not outputs1[0].finished
    assert engine.block_manager.num_free_blocks < initial_free_blocks

    # Decode step (generates token 2 -> reaches max_tokens=2 and finishes)
    outputs2 = engine.step()
    assert len(outputs2) == 1
    assert outputs2[0].finished

    # All blocks returned to free pool immediately
    assert engine.block_manager.num_free_blocks == initial_free_blocks
