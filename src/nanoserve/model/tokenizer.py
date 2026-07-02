"""Tokenizer integration: wraps HuggingFace AutoTokenizer with fallback to SimpleTokenizer."""

from __future__ import annotations

from typing import Protocol


class TokenizerProtocol(Protocol):
    """Protocol for tokenization."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to string."""
        ...

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        ...


class SimpleTokenizer:
    """Byte-level tokenizer for toy models and offline test environments.

    Maps UTF-8 bytes to token IDs modulo vocab_size.
    """

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text as byte values."""
        raw_bytes = text.encode("utf-8")
        return [b % self.vocab_size for b in raw_bytes]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to string."""
        byte_vals = bytes([t % 256 for t in token_ids])
        return byte_vals.decode("utf-8", errors="replace")

    def decode_token(self, token_id: int) -> str:
        """Decode single token ID."""
        return self.decode([token_id])


class HFTokenizer:
    """Wrapper around HuggingFace AutoTokenizer."""

    def __init__(self, hf_tokenizer: object) -> None:
        self._tokenizer = hf_tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode text using HF tokenizer."""
        return list(self._tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs using HF tokenizer."""
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True))  # type: ignore[attr-defined]

    def decode_token(self, token_id: int) -> str:
        """Decode single token ID."""
        return str(self._tokenizer.decode([token_id], skip_special_tokens=False))  # type: ignore[attr-defined]


def get_tokenizer(model_name_or_path: str, vocab_size: int = 256) -> TokenizerProtocol:
    """Get appropriate tokenizer for model_name_or_path."""
    if model_name_or_path == "toy":
        return SimpleTokenizer(vocab_size=vocab_size)

    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        hf_tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return HFTokenizer(hf_tok)
    except Exception:
        # Fallback to SimpleTokenizer if offline or transformers not installed
        return SimpleTokenizer(vocab_size=vocab_size)
