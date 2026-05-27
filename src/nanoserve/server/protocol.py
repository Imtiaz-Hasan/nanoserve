"""Pydantic v2 models matching the OpenAI API schema."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

# --- Request models ---


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""

    model: str = "nanoserve-toy"
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None
    logit_bias: dict[str, float] | None = None

    def get_stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop

    def get_logit_bias(self) -> dict[int, float]:
        if self.logit_bias is None:
            return {}
        return {int(k): v for k, v in self.logit_bias.items()}


class CompletionRequest(BaseModel):
    """POST /v1/completions request body."""

    model: str = "nanoserve-toy"
    prompt: str
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None

    def get_stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return self.stop


# --- Response models ---


class UsageInfo(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Non-streaming chat completion response."""

    id: str = ""
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nanoserve-toy"
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """Incremental message content for streaming."""

    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    """A single choice in a streaming chunk."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """SSE streaming chunk for chat completions."""

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nanoserve-toy"
    choices: list[ChatCompletionStreamChoice]


class CompletionChoice(BaseModel):
    """A single choice in a completion response."""

    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    """Non-streaming completion response."""

    id: str = ""
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "nanoserve-toy"
    choices: list[CompletionChoice]
    usage: UsageInfo


# --- Model listing ---


class ModelCard(BaseModel):
    """A model available for inference."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nanoserve"


class ModelList(BaseModel):
    """List of available models."""

    object: str = "list"
    data: list[ModelCard]


# --- Health ---


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model: str = ""
    num_waiting: int = 0
    num_running: int = 0
