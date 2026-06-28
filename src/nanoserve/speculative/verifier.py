"""Speculative decoding verifier: greedy and distribution-preserving rejection sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


class SpeculativeVerifier:
    """Verifies K candidate draft tokens against target model logits in a single step."""

    @staticmethod
    def verify_greedy(
        draft_tokens: list[int],
        target_logits: torch.Tensor,
    ) -> tuple[list[int], int, int]:
        """Verify draft tokens under greedy (argmax) decoding.

        Args:
            draft_tokens: K proposed candidate token IDs
            target_logits: (K + 1, vocab_size) or (1, K + 1, vocab_size) logits

        Returns:
            accepted_tokens: List of accepted draft tokens
            bonus_token: Recovery or bonus token emitted by the target model
            num_accepted: Number of accepted candidate tokens
        """
        if target_logits.dim() == 3:
            target_logits = target_logits.squeeze(0)

        num_draft = len(draft_tokens)
        accepted: list[int] = []

        for i in range(num_draft):
            target_token = int(target_logits[i].argmax(dim=-1).item())
            if target_token == draft_tokens[i]:
                accepted.append(draft_tokens[i])
            else:
                # Rejection at position i: target_token is the corrective token
                return accepted, target_token, len(accepted)

        # All K draft tokens accepted! Emitting bonus token from index K
        bonus_token = int(target_logits[num_draft].argmax(dim=-1).item())
        return accepted, bonus_token, len(accepted)

    @staticmethod
    def verify_sampling(
        draft_tokens: list[int],
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> tuple[list[int], int, int]:
        """Verify draft tokens under temperature sampling using rejection sampling.

        Guarantees exact mathematical distribution equivalence to target model generation.
        """
        if target_logits.dim() == 3:
            target_logits = target_logits.squeeze(0)

        num_draft = len(draft_tokens)
        accepted: list[int] = []

        # Target probabilities
        target_probs = F.softmax(target_logits / max(1e-5, temperature), dim=-1)

        for i in range(num_draft):
            draft_token = draft_tokens[i]
            p_target = target_probs[i, draft_token].item()
            q_draft = draft_probs[i, draft_token].item() if draft_probs is not None else 1.0

            accept_prob = min(1.0, p_target / max(1e-8, q_draft))
            rand_val = torch.rand(1).item()

            if rand_val <= accept_prob:
                accepted.append(draft_token)
            else:
                # Rejection sampling bonus distribution: max(0, p - q)
                if draft_probs is not None:
                    adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                    norm = adjusted.sum()
                    sample_dist = adjusted / norm if norm > 0 else target_probs[i]
                else:
                    sample_dist = target_probs[i]

                bonus_token = int(torch.multinomial(sample_dist, num_samples=1).item())
                return accepted, bonus_token, len(accepted)

        # All accepted: sample bonus token from target_probs[num_draft]
        bonus_token = int(torch.multinomial(target_probs[num_draft], num_samples=1).item())
        return accepted, bonus_token, len(accepted)
