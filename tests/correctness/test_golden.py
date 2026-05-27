"""Golden test: greedy generation with the toy model is deterministic."""

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams


def test_greedy_deterministic() -> None:
    """Two runs with the same seed produce identical output."""
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=16),
        scheduler=SchedulerConfig(max_num_seqs=1),
        device="cpu",
        seed=42,
    )

    results: list[list[int]] = []

    for _ in range(2):
        engine = LLMEngine(config)
        engine.add_request(
            "golden-test",
            "Hello",
            SamplingParams(temperature=0.0, max_tokens=16),
        )

        output_tokens: list[int] = []
        while engine.has_unfinished_requests():
            outputs = engine.step()
            for out in outputs:
                for comp in out.outputs:
                    output_tokens.append(comp.token_id)

        results.append(output_tokens)

    assert len(results[0]) == 16
    assert results[0] == results[1], "Greedy generation must be deterministic across runs"


def test_golden_multiple_prompts() -> None:
    """Multiple prompts produce non-empty, consistent greedy output."""
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=128, block_size=16),
        scheduler=SchedulerConfig(max_num_seqs=8),
        device="cpu",
        seed=42,
    )

    prompts = [
        "What is machine learning?",
        "Explain transformers.",
        "Write a function.",
        "The capital of France is",
        "Hello world",
    ]

    engine = LLMEngine(config)
    for i, prompt in enumerate(prompts):
        engine.add_request(
            f"golden-{i}",
            prompt,
            SamplingParams(temperature=0.0, max_tokens=8),
        )

    finished: dict[str, list[int]] = {}
    while engine.has_unfinished_requests():
        outputs = engine.step()
        for out in outputs:
            if out.request_id not in finished:
                finished[out.request_id] = []
            for comp in out.outputs:
                finished[out.request_id].append(comp.token_id)

    # All prompts should have produced exactly 8 tokens
    for req_id, tokens in finished.items():
        assert len(tokens) == 8, f"{req_id} produced {len(tokens)} tokens, expected 8"
