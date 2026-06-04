"""Statistical correctness tests for sampler probability mass and temperature properties."""

from collections import Counter

import torch

from nanoserve.sampling.params import SamplingParams
from nanoserve.sampling.sampler import Sampler


def test_top_p_nucleus_mass_confinement() -> None:
    """Statistical correctness: top-p=0.5 never selects tokens outside the top 50% mass."""
    sampler = Sampler()

    # Create logits where tokens [0, 1] make up >60% of probability mass
    logits = torch.tensor([[4.0, 3.5, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0]])
    probs = torch.softmax(logits, dim=-1)[0]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    allowed_indices = set()
    for idx, cumsum in zip(sorted_indices, cumulative_probs, strict=False):
        allowed_indices.add(int(idx.item()))
        if cumsum.item() >= 0.5:
            break

    for i in range(200):
        # vary seed
        params_with_seed = SamplingParams(temperature=1.0, top_p=0.5, seed=1000 + i)
        token = sampler.sample(logits, [params_with_seed])[0]
        assert token in allowed_indices, (
            f"Token {token} selected outside top-p allowed set {allowed_indices}"
        )


def test_low_temperature_argmax_convergence() -> None:
    """As temperature approaches zero, stochastic sampling converges to argmax."""
    sampler = Sampler()
    logits = torch.tensor([[1.0, 2.5, 2.49, -1.0]])  # Token 1 is highest

    for i in range(50):
        params = SamplingParams(temperature=0.001, seed=i)
        token = sampler.sample(logits, [params])[0]
        assert token == 1, f"Low-temperature sample {token} did not match argmax 1"


def test_high_temperature_uniform_convergence() -> None:
    """As temperature -> infinity, probability distribution converges to uniform across tokens."""
    sampler = Sampler()
    # Distinct logits across 4 tokens
    logits = torch.tensor([[10.0, 5.0, 1.0, -2.0]])

    num_samples = 2000
    counts: Counter[int] = Counter()
    for i in range(num_samples):
        params_i = SamplingParams(temperature=500.0, seed=i)
        token = sampler.sample(logits, [params_i])[0]
        counts[token] += 1

    # In uniform distribution across 4 tokens, each token expects ~25% (500 samples)
    for token_id in range(4):
        freq = counts[token_id] / num_samples
        msg = f"Freq {freq:.3f} for token {token_id} deviated from uniform ~0.25"
        assert 0.20 <= freq <= 0.30, msg
