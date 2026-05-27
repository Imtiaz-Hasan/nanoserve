"""Engine output types: per-request and per-sequence completion results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompletionOutput:
    """A single generated token or completed generation."""

    index: int = 0
    token_id: int = 0
    text: str = ""
    finish_reason: str | None = None
    logprobs: dict[int, float] | None = None


@dataclass
class RequestOutput:
    """Output for one step of one request."""

    request_id: str
    prompt: str = ""
    outputs: list[CompletionOutput] = field(default_factory=list)
    finished: bool = False
    prompt_token_ids: list[int] = field(default_factory=list)
    num_output_tokens: int = 0
