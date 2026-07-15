"""Unit tests for constrained guided decoding (Regex and JSON Schema logit processors)."""

import torch

from nanoserve.model.tokenizer import SimpleTokenizer
from nanoserve.sampling.constrained import JsonSchemaLogitProcessor, RegexLogitProcessor


def test_regex_logit_processor_digit_masking() -> None:
    """Verify RegexLogitProcessor masks non-digit tokens when enforcing digits."""
    tokenizer = SimpleTokenizer(vocab_size=256)
    # Digits are ASCII 48 ('0') to 57 ('9')
    processor = RegexLogitProcessor(regex_pattern=r"^[0-9]+$", tokenizer=tokenizer, vocab_size=256)

    logits = torch.zeros(256)
    current_text = "123"

    masked = processor(current_text, logits)

    # Valid next tokens (digits) should have normal logits
    for digit_ascii in range(ord("0"), ord("9") + 1):
        assert masked[digit_ascii].item() == 0.0

    # Non-digit letters (e.g. 'a', 'Z', '!') should be masked with -inf
    assert masked[ord("a")].item() == -float("inf")
    assert masked[ord("Z")].item() == -float("inf")
    assert masked[ord("!")].item() == -float("inf")


def test_json_schema_initial_token_masking() -> None:
    """Verify JsonSchemaLogitProcessor forces opening brace or bracket on empty prefix."""
    tokenizer = SimpleTokenizer(vocab_size=256)
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    processor = JsonSchemaLogitProcessor(schema=schema, tokenizer=tokenizer, vocab_size=256)

    logits = torch.zeros(256)
    masked = processor("", logits)

    # Opening brace should be valid
    assert masked[ord("{")].item() == 0.0
    # Random letter should be masked
    assert masked[ord("a")].item() == -float("inf")
    assert masked[ord("x")].item() == -float("inf")


def test_json_schema_completion_masking() -> None:
    """Verify JsonSchemaLogitProcessor suppresses non-whitespace once valid JSON is closed."""
    tokenizer = SimpleTokenizer(vocab_size=256)
    schema = {"type": "object"}
    processor = JsonSchemaLogitProcessor(schema=schema, tokenizer=tokenizer, vocab_size=256)

    logits = torch.zeros(256)
    complete_json = '{"name": "nanoserve"}'

    masked = processor(complete_json, logits)

    # Alphabet characters should be masked to terminate or allow whitespace
    assert masked[ord("a")].item() == -float("inf")
    assert masked[ord("1")].item() == -float("inf")
