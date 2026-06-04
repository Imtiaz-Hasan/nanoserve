"""Unit tests for the sampling suite: penalties, seeded RNG, top-k/p, min-p, stop strings."""

import torch

from nanoserve.sampling.params import SamplingParams
from nanoserve.sampling.penalties import (
    apply_frequency_presence_penalties,
    apply_repetition_penalty,
)
from nanoserve.sampling.sampler import Sampler
from nanoserve.sampling.stop import StopChecker


def test_repetition_penalty_math() -> None:
    """Verify repetition penalty scales positive logits down and negative logits negative."""
    logits = torch.tensor([2.0, -2.0, 4.0, 0.0])
    token_ids = [0, 1]  # penalize tokens 0 and 1

    penalized = apply_repetition_penalty(logits.clone(), token_ids, penalty=2.0)

    assert penalized[0].item() == 1.0  # 2.0 / 2.0
    assert penalized[1].item() == -4.0  # -2.0 * 2.0
    assert penalized[2].item() == 4.0  # unpenalized
    assert penalized[3].item() == 0.0


def test_frequency_and_presence_penalties_math() -> None:
    """Verify additive frequency and presence penalties."""
    logits = torch.tensor([5.0, 5.0, 5.0])
    output_tokens = [0, 0, 0, 1]  # token 0 appears 3 times, token 1 appears 1 time

    penalized = apply_frequency_presence_penalties(
        logits=logits.clone(),
        output_token_ids=output_tokens,
        frequency_penalty=0.5,
        presence_penalty=1.0,
    )

    # Token 0: 5.0 - (0.5 * 3 + 1.0) = 5.0 - 2.5 = 2.5
    assert abs(penalized[0].item() - 2.5) < 1e-5
    # Token 1: 5.0 - (0.5 * 1 + 1.0) = 5.0 - 1.5 = 3.5
    assert abs(penalized[1].item() - 3.5) < 1e-5
    # Token 2: unpenalized
    assert penalized[2].item() == 5.0


def test_seeded_reproducibility() -> None:
    """Verify two runs with the same seed generate identical stochastic token sequences."""
    sampler = Sampler()
    logits = torch.randn(1, 100)
    params1 = SamplingParams(temperature=1.2, top_k=20, top_p=0.9, seed=1337)
    params2 = SamplingParams(temperature=1.2, top_k=20, top_p=0.9, seed=1337)

    sampled1 = [sampler.sample(logits, [params1])[0] for _ in range(10)]
    sampled2 = [sampler.sample(logits, [params2])[0] for _ in range(10)]

    assert sampled1 == sampled2


def test_top_k_truncation() -> None:
    """Verify top-k=2 restricts selection exclusively to the two highest logits."""
    sampler = Sampler()
    # Logits where tokens 5 and 8 are vastly higher
    logits = torch.zeros(1, 10)
    logits[0, 5] = 10.0
    logits[0, 8] = 9.0

    params = SamplingParams(temperature=1.0, top_k=2, seed=42)

    sampled_tokens = {sampler.sample(logits, [params])[0] for _ in range(50)}
    assert sampled_tokens.issubset({5, 8})


def test_min_p_relative_filtering() -> None:
    """Verify min-p filters out tokens whose probability is below min_p * max_p."""
    sampler = Sampler()
    logits = torch.zeros(1, 10)
    logits[0, 0] = 5.0  # Dominant token (~90% probability)
    logits[0, 1] = 2.0  # Secondary token (~5% probability)
    logits[0, 2] = -5.0  # Tiny token (<0.01% probability)

    # min_p = 0.2: tokens with p < 0.2 * p_max will be dropped
    params = SamplingParams(temperature=1.0, min_p=0.2, seed=42)

    for _ in range(30):
        token = sampler.sample(logits, [params])[0]
        assert token != 2, "Token with probability below min_p relative threshold was selected"


def test_stop_string_unicode_and_multibyte() -> None:
    """Verify stop checker correctly catches multi-byte UTF-8 emojis and strings."""
    stop_checker = StopChecker(stop_strings=["🤖", "🚀", "STOP"], stop_token_ids=[])

    assert stop_checker.should_stop_string("Generating text with 🤖") == "🤖"
    assert stop_checker.should_stop_string("Liftoff 🚀") == "🚀"
    assert stop_checker.should_stop_string("Regular text") is None


def test_stop_string_overlapping_prefixes() -> None:
    """Verify stop checker handles overlapping prefix stop strings."""
    stop_checker = StopChecker(stop_strings=["foo", "foobar"], stop_token_ids=[])

    assert stop_checker.should_stop_string("Test foobar") == "foobar"
    assert stop_checker.should_stop_string("Test foo") == "foo"
