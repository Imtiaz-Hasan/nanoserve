"""Penalties for LLM token generation: repetition, frequency, and presence penalties."""

from __future__ import annotations

from collections import Counter

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    token_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Apply multiplicative repetition penalty (Keskar et al., 2019).

    For tokens that have appeared in the prompt or previously generated tokens:
    - If logit > 0: logit = logit / penalty
    - If logit <= 0: logit = logit * penalty

    Args:
        logits: (vocab_size,) 1D float tensor
        token_ids: list of token IDs to penalize
        penalty: multiplicative factor (penalty >= 1.0; 1.0 means no penalty)

    Returns:
        Modified logits tensor.
    """
    if penalty == 1.0 or not token_ids:
        return logits

    unique_tokens = set(token_ids)
    vocab_size = logits.shape[0]

    for token_id in unique_tokens:
        if 0 <= token_id < vocab_size:
            val = logits[token_id].item()
            if val > 0:
                logits[token_id] = val / penalty
            else:
                logits[token_id] = val * penalty

    return logits


def apply_frequency_presence_penalties(
    logits: torch.Tensor,
    output_token_ids: list[int],
    frequency_penalty: float,
    presence_penalty: float,
) -> torch.Tensor:
    """Apply additive frequency and presence penalties (OpenAI schema).

    Formula: logit_i = logit_i - (frequency_penalty * count_i + presence_penalty * (count_i > 0))

    Args:
        logits: (vocab_size,) 1D float tensor
        output_token_ids: list of generated output token IDs
        frequency_penalty: penalty proportional to token occurrence count
        presence_penalty: flat penalty for any token that has appeared at least once

    Returns:
        Modified logits tensor.
    """
    if (frequency_penalty == 0.0 and presence_penalty == 0.0) or not output_token_ids:
        return logits

    counts = Counter(output_token_ids)
    vocab_size = logits.shape[0]

    for token_id, count in counts.items():
        if 0 <= token_id < vocab_size:
            penalty = frequency_penalty * count + presence_penalty
            logits[token_id] -= penalty

    return logits
