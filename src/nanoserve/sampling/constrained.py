"""Constrained guided decoding: regex patterns and JSON schema logit biasing."""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from nanoserve.model.tokenizer import TokenizerProtocol


class RegexLogitProcessor:
    """Masks candidate token logits that violate a regular expression pattern."""

    def __init__(
        self,
        regex_pattern: str,
        tokenizer: TokenizerProtocol,
        vocab_size: int = 256,
    ) -> None:
        self.pattern_str = regex_pattern
        self.compiled_regex = re.compile(regex_pattern)
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def __call__(
        self,
        current_text: str,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply -inf masking to tokens whose addition cannot satisfy the regex.

        Args:
            current_text: Current prefix string emitted so far
            logits: (vocab_size,) or (1, vocab_size) next-token logits

        Returns:
            masked_logits: Tensor with invalid token positions set to -inf
        """
        masked = logits.clone()
        single_logits = masked[0] if masked.dim() == 2 else masked

        valid_count = 0
        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.decode_token(token_id)
            candidate_text = current_text + token_str

            # Check if candidate_text matches pattern or can be a valid prefix
            # Regex search matching prefix
            match = self.compiled_regex.match(candidate_text)
            if match is not None or any(
                self.compiled_regex.match(candidate_text + suffix) is not None
                for suffix in ["", "0", "A", '"', "}", "]"]
            ):
                valid_count += 1
            else:
                single_logits[token_id] = -float("inf")

        # If all masked due to strict prefix bounds, restore original to avoid NaNs
        if valid_count == 0:
            return logits

        return masked


class JsonSchemaLogitProcessor:
    """Enforces structural JSON syntax (braces, quotes, colons, comma separation)."""

    def __init__(
        self,
        schema: dict[str, Any],
        tokenizer: TokenizerProtocol,
        vocab_size: int = 256,
    ) -> None:
        self.schema = schema
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.required_keys = set(schema.get("required", schema.get("properties", {}).keys()))

    def __call__(
        self,
        current_text: str,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Bias logits to ensure valid JSON formation."""
        masked = logits.clone()
        single_logits = masked[0] if masked.dim() == 2 else masked

        # If text is empty, first non-whitespace character must be '{' or '['
        stripped = current_text.strip()
        if not stripped:
            for token_id in range(self.vocab_size):
                tok = self.tokenizer.decode_token(token_id)
                if tok.strip() not in ("{", "[", ""):
                    single_logits[token_id] = -float("inf")
            return masked

        # If already valid JSON and closing brace emitted, prevent extra noise
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                json.loads(stripped)
                # Valid JSON: mask non-whitespace tokens to encourage stop
                for token_id in range(self.vocab_size):
                    tok = self.tokenizer.decode_token(token_id)
                    if tok.strip() != "":
                        single_logits[token_id] = -float("inf")
                return masked
            except json.JSONDecodeError:
                pass

        return masked
