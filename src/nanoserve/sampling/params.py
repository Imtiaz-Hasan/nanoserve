"""Sampling parameters with validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Per-request sampling configuration.

    Week 1: only temperature=0.0 (greedy) and max_tokens are functional.
    Week 3 activates the full suite: top-k, top-p, min-p, penalties.
    """

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    seed: int | None = None
    logit_bias: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.temperature < 0.0:
            msg = f"temperature must be >= 0, got {self.temperature}"
            raise ValueError(msg)
        if self.top_p < 0.0 or self.top_p > 1.0:
            msg = f"top_p must be in [0, 1], got {self.top_p}"
            raise ValueError(msg)
        if self.min_p < 0.0 or self.min_p > 1.0:
            msg = f"min_p must be in [0, 1], got {self.min_p}"
            raise ValueError(msg)
        if self.max_tokens < 1:
            msg = f"max_tokens must be >= 1, got {self.max_tokens}"
            raise ValueError(msg)
        if self.repetition_penalty < 1.0:
            msg = f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}"
            raise ValueError(msg)

    @property
    def is_greedy(self) -> bool:
        """Whether this is greedy decoding (temperature == 0)."""
        return self.temperature == 0.0
