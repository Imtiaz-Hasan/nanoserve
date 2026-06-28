"""Speculative decoding proposers: N-gram prompt-lookup and lightweight draft model."""

from __future__ import annotations

from typing import Protocol

import torch

from nanoserve.core.sequence import Sequence
from nanoserve.model.llama import LlamaForCausalLM


class SpeculativeProposer(Protocol):
    """Protocol for proposing speculative candidate tokens."""

    def propose(self, seq: Sequence, num_tokens: int) -> list[int]:
        """Propose up to `num_tokens` candidate token IDs for the next decode steps."""
        ...


class NgramProposer:
    """Prompt-lookup proposer based on N-gram matching across sequence token history.

    Scans the prompt and generated history for occurrences of the trailing N-gram.
    If an exact match is found earlier in the sequence, proposes the following K tokens.
    Requires zero model parameters and zero GPU memory.
    """

    def __init__(self, ngram_size: int = 3) -> None:
        self.ngram_size = max(1, ngram_size)

    def propose(self, seq: Sequence, num_tokens: int) -> list[int]:
        """Find matching N-gram in sequence history and return candidate draft tokens."""
        all_tokens = seq.all_token_ids
        if len(all_tokens) < self.ngram_size + 1:
            return []

        # Try from ngram_size down to 1
        for n in range(self.ngram_size, 0, -1):
            if len(all_tokens) < n + 1:
                continue

            query_ngram = all_tokens[-n:]
            # Search earlier in the sequence (excluding the trailing query ngram itself)
            search_window = all_tokens[:-1]
            search_len = len(search_window)

            # Search backwards from recent history to favor local repetition
            for i in range(search_len - n, -1, -1):
                if search_window[i : i + n] == query_ngram:
                    # Found match: grab following tokens up to num_tokens
                    match_end = i + n
                    draft = search_window[match_end : match_end + num_tokens]
                    if draft:
                        return draft

        return []


class DraftModelProposer:
    """Speculative proposer using a lightweight draft language model."""

    def __init__(self, draft_model: LlamaForCausalLM, device: str = "cpu") -> None:
        self.draft_model = draft_model
        self.device = device

    def propose(self, seq: Sequence, num_tokens: int) -> list[int]:
        """Autoregressively generate candidate tokens using the draft model."""
        if num_tokens <= 0:
            return []

        draft_tokens: list[int] = []
        cur_tokens = list(seq.all_token_ids)

        for _ in range(num_tokens):
            input_ids = torch.tensor([cur_tokens], dtype=torch.long, device=self.device)
            positions = torch.tensor(
                list(range(len(cur_tokens))), dtype=torch.long, device=self.device
            )

            with torch.no_grad():
                logits, _ = self.draft_model.forward(input_ids=input_ids, positions=positions)

            next_token = int(logits[0, -1, :].argmax().item())
            draft_tokens.append(next_token)
            cur_tokens.append(next_token)

        return draft_tokens
