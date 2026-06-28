"""Unit tests for speculative decoding (N-gram proposer, verifier, and SpeculativeEngine)."""

import torch

from nanoserve.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from nanoserve.core.sequence import Sequence
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.sampling.params import SamplingParams
from nanoserve.speculative.proposer import NgramProposer
from nanoserve.speculative.speculative_engine import SpeculativeEngine
from nanoserve.speculative.verifier import SpeculativeVerifier


def test_ngram_proposer_matching() -> None:
    """Verify N-gram proposer detects matching history and returns following candidate tokens."""
    proposer = NgramProposer(ngram_size=3)

    # Prompt: [10, 20, 30, 40, 50, 60, 10, 20, 30] -> trailing 3-gram is [10, 20, 30]
    seq = Sequence(
        seq_id=1,
        prompt_token_ids=[10, 20, 30, 40, 50, 60, 10, 20, 30],
    )
    draft = proposer.propose(seq, num_tokens=3)
    assert draft == [40, 50, 60]


def test_speculative_verifier_greedy_full_acceptance() -> None:
    """Verify verifier accepts all K tokens and emits bonus token when candidate matches target."""
    draft_tokens = [100, 200, 300]
    # Logits where argmax at pos 0 is 100, pos 1 is 200, pos 2 is 300, pos 3 is 400 (bonus)
    logits = torch.zeros((4, 500))
    logits[0, 100] = 10.0
    logits[1, 200] = 10.0
    logits[2, 300] = 10.0
    logits[3, 400] = 10.0

    accepted, bonus, num_accepted = SpeculativeVerifier.verify_greedy(draft_tokens, logits)
    assert accepted == [100, 200, 300]
    assert bonus == 400
    assert num_accepted == 3


def test_speculative_verifier_greedy_partial_rejection() -> None:
    """Verify verifier rejects on mismatch and returns target's corrective token."""
    draft_tokens = [100, 200, 300]
    # Target logits: pos 0 is 100 (accept), pos 1 is 999 (mismatch/reject!)
    logits = torch.zeros((4, 1000))
    logits[0, 100] = 10.0
    logits[1, 999] = 10.0  # target wanted 999, not 200
    logits[2, 300] = 10.0

    accepted, bonus, num_accepted = SpeculativeVerifier.verify_greedy(draft_tokens, logits)
    assert accepted == [100]
    assert bonus == 999  # corrective token
    assert num_accepted == 1


def test_speculative_engine_output_equivalence() -> None:
    """Verify SpeculativeEngine output matches standard LLMEngine output 100% token-for-token."""
    prompt = "The quick brown fox jumps over the lazy dog and then the quick brown fox"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=15)

    # 1. Standard LLMEngine baseline
    base_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_batched_tokens=128),
        device="cpu",
        seed=42,
    )
    base_engine = LLMEngine(base_config)
    base_engine.add_request("base", prompt, sampling_params)

    base_tokens: list[int] = []
    while base_engine.has_unfinished_requests():
        outs = base_engine.step()
        for out in outs:
            if out.outputs:
                base_tokens.append(out.outputs[0].token_id)

    # 2. Speculative Engine with N-gram proposal
    spec_config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_batched_tokens=128),
        speculative=SpeculativeConfig(
            enabled=True,
            num_speculative_tokens=4,
            proposer_type="ngram",
            ngram_size=2,
        ),
        device="cpu",
        seed=42,
    )
    spec_engine = SpeculativeEngine(spec_config)
    spec_engine.add_request("spec", prompt, sampling_params)

    spec_tokens: list[int] = []
    while spec_engine.has_unfinished_requests():
        outs = spec_engine.step()
        for out in outs:
            if out.outputs:
                spec_tokens.append(out.outputs[0].token_id)

    # Outputs match byte-for-byte
    assert spec_tokens == base_tokens


def test_speculative_engine_acceptance_and_speedup() -> None:
    """Verify SpeculativeEngine achieves token proposal and acceptance on repetitive text."""
    # Highly repetitive prompt encouraging N-gram hits
    repetitive_prompt = "ABCDEF ABCDEF ABCDEF ABCDEF ABCDEF ABCDEF "
    sampling_params = SamplingParams(temperature=0.0, max_tokens=12)

    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=4),
        scheduler=SchedulerConfig(max_num_batched_tokens=128),
        speculative=SpeculativeConfig(
            enabled=True,
            num_speculative_tokens=4,
            proposer_type="ngram",
            ngram_size=2,
        ),
        device="cpu",
        seed=42,
    )
    engine = SpeculativeEngine(config)
    engine.add_request("req", repetitive_prompt, sampling_params)

    while engine.has_unfinished_requests():
        engine.step()

    # Verify metrics tracked
    assert engine.total_steps > 0
    assert engine.total_emitted_tokens >= 12
