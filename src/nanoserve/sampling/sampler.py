"""Token sampler: greedy argmax and (future) stochastic sampling."""

from __future__ import annotations

import torch

from nanoserve.sampling.params import SamplingParams


class Sampler:
    """Batched token sampling from logits.

    Week 1: greedy argmax only.
    Week 3: temperature, top-k, top-p, min-p, penalties, seeded RNG.
    """

    def sample(
        self,
        logits: torch.Tensor,
        sampling_params_list: list[SamplingParams],
    ) -> list[int]:
        """Sample one token per sequence in the batch.

        Args:
            logits: (batch, vocab_size) — logits for the last position of each sequence
            sampling_params_list: per-sequence sampling parameters

        Returns:
            List of sampled token ids, one per sequence.
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
            token_id = self._sample_single(logits[i], sampling_params_list[i])
            results.append(token_id)
        return results

    def _sample_single(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
    ) -> int:
        """Sample a single token from logits.

        Week 1: greedy only. All temperature > 0 falls back to argmax.
        """
        # Apply logit bias
        if params.logit_bias:
            for token_id, bias in params.logit_bias.items():
                if 0 <= token_id < logits.shape[0]:
                    logits[token_id] += bias

        if params.is_greedy or params.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())

        # Week 3: full stochastic sampling pipeline
        # For now, fall back to argmax
        return int(logits.argmax(dim=-1).item())
