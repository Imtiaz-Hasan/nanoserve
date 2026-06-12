"""Unit tests for preemption mechanisms: recompute, CPU memory swap, and output determinism."""

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams


def test_recompute_preemption_under_heavy_pressure() -> None:
    """Verify engine completes requests via recompute preemption under tight memory constraints."""
    # 4 blocks × 4 tokens = 16 tokens maximum GPU capacity
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=4, block_size=4),
        scheduler=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            preemption_mode="recompute",
        ),
        device="cpu",
        seed=42,
    )
    engine = LLMEngine(config)

    # 6 requests × 6 tokens generated each = 36 tokens needed (> 16 block capacity)
    num_requests = 6
    for i in range(num_requests):
        engine.add_request(
            f"req-{i}",
            f"P{i}",
            SamplingParams(temperature=0.0, max_tokens=6),
        )

    completed_reqs: set[str] = set()
    step_count = 0

    while engine.has_unfinished_requests() and step_count < 100:
        step_count += 1
        outputs = engine.step()
        for out in outputs:
            if out.finished:
                completed_reqs.add(out.request_id)

    # All requests must complete successfully without crashing
    assert len(completed_reqs) == num_requests
    assert engine.scheduler.num_preemptions > 0
    assert engine.block_manager.num_free_blocks == 4


def test_swap_preemption_with_cpu_memory() -> None:
    """Verify engine completes requests via host CPU swap preemption."""
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=4, num_cpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=2048,
            preemption_mode="swap",
        ),
        device="cpu",
        seed=42,
    )
    engine = LLMEngine(config)

    num_requests = 5
    for i in range(num_requests):
        engine.add_request(
            f"swap-req-{i}",
            f"P{i}",
            SamplingParams(temperature=0.0, max_tokens=6),
        )

    completed_reqs: set[str] = set()
    step_count = 0

    while engine.has_unfinished_requests() and step_count < 100:
        step_count += 1
        outputs = engine.step()
        for out in outputs:
            if out.finished:
                completed_reqs.add(out.request_id)

    assert len(completed_reqs) == num_requests
    assert engine.scheduler.num_preemptions > 0
    assert engine.block_manager.num_free_blocks == 4


def test_preemption_output_determinism() -> None:
    """Verify preempted requests produce identical outputs to non-preempted reference runs."""
    prompt = "Test prompt"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=8)

    # 1. Reference run with abundant memory (no preemption)
    ref_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(preemption_mode="recompute"),
        device="cpu",
        seed=42,
    )
    ref_engine = LLMEngine(ref_config)
    ref_engine.add_request("target", prompt, sampling_params)

    ref_output_tokens: list[int] = []
    while ref_engine.has_unfinished_requests():
        outputs = ref_engine.step()
        for out in outputs:
            if out.outputs:
                ref_output_tokens.append(out.outputs[0].token_id)

    # 2. Constrained run forcing preemption of the target request
    constrained_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=6, block_size=4),
        scheduler=SchedulerConfig(max_num_seqs=4, preemption_mode="recompute"),
        device="cpu",
        seed=42,
    )
    constrained_engine = LLMEngine(constrained_config)

    # Comp-1 (3 blocks) + target (3 blocks) uses all 6 GPU blocks
    constrained_engine.add_request("comp-1", "Comp prompt", SamplingParams(max_tokens=6))
    constrained_engine.add_request("target", prompt, sampling_params)

    preempted_output_tokens: list[int] = []
    step_count = 0
    while constrained_engine.has_unfinished_requests() and step_count < 50:
        step_count += 1
        outputs = constrained_engine.step()
        for out in outputs:
            if out.request_id == "target" and out.outputs:
                preempted_output_tokens.append(out.outputs[0].token_id)

    assert constrained_engine.scheduler.num_preemptions > 0
    assert preempted_output_tokens == ref_output_tokens
