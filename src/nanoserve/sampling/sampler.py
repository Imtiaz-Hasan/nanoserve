"""Token sampler: greedy, temperature scaling, top-k, top-p, min-p, and seeded RNG."""

from __future__ import annotations

import torch

from nanoserve.sampling.params import SamplingParams
from nanoserve.sampling.penalties import (
    apply_frequency_presence_penalties,
    apply_repetition_penalty,
)


class Sampler:
    """Batched token sampling from logits supporting diverse per-sequence strategies.

    Supports greedy argmax, temperature scaling, top-k truncation, top-p (nucleus),
    min-p relative filtering, repetition/frequency/presence penalties, and isolated
    per-request seeded generators for 100% deterministic reproducibility.
    """

    def __init__(self) -> None:
        self._generators: dict[int, torch.Generator] = {}

    def sample(
        self,
        logits: torch.Tensor,
        sampling_params_list: list[SamplingParams],
        history_tokens_list: list[list[int]] | None = None,
        output_tokens_list: list[list[int]] | None = None,
    ) -> list[int]:
        """Sample one token per sequence in the batch.

        Args:
            logits: (batch_size, vocab_size) float tensor
            sampling_params_list: list of per-sequence SamplingParams
            history_tokens_list: list of all tokens (prompt + output) per sequence
            output_tokens_list: list of generated output tokens per sequence

        Returns:
            List of sampled token IDs, one per sequence.
        """
        batch_size = logits.shape[0]
        if batch_size != len(sampling_params_list):
            msg = (
                f"Batch size mismatch: logits has {batch_size} rows "
                f"but got {len(sampling_params_list)} sampling params"
            )
            raise ValueError(msg)

        results: list[int] = []
        for i in range(batch_size):
            params = sampling_params_list[i]
            history = history_tokens_list[i] if history_tokens_list is not None else []
            output_tokens = output_tokens_list[i] if output_tokens_list is not None else []

            token_id = self._sample_single(
                logits=logits[i].clone(),
                params=params,
                history_tokens=history,
                output_tokens=output_tokens,
            )
            results.append(token_id)

        return results

    def _sample_single(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        history_tokens: list[int],
        output_tokens: list[int],
    ) -> int:
        """Sample a single token from 1D logits tensor."""
        # 1. Apply logit bias
        if params.logit_bias:
            for token_id, bias in params.logit_bias.items():
                if 0 <= token_id < logits.shape[0]:
                    logits[token_id] += bias

        # 2. Apply repetition penalty
        if params.repetition_penalty != 1.0 and history_tokens:
            logits = apply_repetition_penalty(logits, history_tokens, params.repetition_penalty)

        # 3. Apply frequency and presence penalties
        if (params.frequency_penalty != 0.0 or params.presence_penalty != 0.0) and output_tokens:
            logits = apply_frequency_presence_penalties(
                logits, output_tokens, params.frequency_penalty, params.presence_penalty
            )

        # 4. Greedy sampling (temperature == 0.0)
        if params.is_greedy or params.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())

        # 5. Temperature scaling
        scaled_logits = logits / max(params.temperature, 1e-5)

        # 6. Top-K truncation
        if params.top_k > 0 and params.top_k < scaled_logits.shape[0]:
            top_k_val = torch.topk(scaled_logits, params.top_k).values[-1]
            scaled_logits = torch.where(
                scaled_logits < top_k_val,
                torch.tensor(float("-inf"), dtype=scaled_logits.dtype, device=scaled_logits.device),
                scaled_logits,
            )

        # 7. Compute probabilities
        probs = torch.softmax(scaled_logits, dim=-1)

        # 8. Top-P (nucleus) filtering
        if params.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens with cumulative mass above top_p (preserving at least 1 token)
            sorted_indices_to_remove = cumulative_probs > params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter mask back
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)

        # 9. Min-P filtering
        if params.min_p > 0.0:
            max_prob = probs.max().item()
            threshold = params.min_p * max_prob
            probs = torch.where(
                probs < threshold,
                torch.zeros_like(probs),
                probs,
            )

        # Re-normalize probabilities
        prob_sum = probs.sum()
        if prob_sum == 0.0 or torch.isnan(prob_sum):
            # Fallback to argmax if probabilities collapsed
            return int(logits.argmax(dim=-1).item())

        probs = probs / prob_sum

        # 10. Sample with seeded generator if seed provided
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=probs.device)
            generator.manual_seed(params.seed)

        sampled_token = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(sampled_token.item())
