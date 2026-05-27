"""OpenAI-compatible API routes: /v1/chat/completions, /v1/completions, /v1/models, /health."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from nanoserve.engine.async_engine import AsyncLLMEngine
from nanoserve.sampling.params import SamplingParams
from nanoserve.server.protocol import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    HealthResponse,
    ModelCard,
    ModelList,
    UsageInfo,
)

router = APIRouter()


def _get_engine(request: Request) -> AsyncLLMEngine:
    """Retrieve the engine from the app state."""
    engine: AsyncLLMEngine = request.app.state.engine
    return engine


def _get_model_name(request: Request) -> str:
    """Retrieve the model name from the app state."""
    name: str = request.app.state.model_name
    return name


# --- Health ---


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    engine = _get_engine(request)
    scheduler = engine.engine.scheduler
    return HealthResponse(
        status="ok",
        model=_get_model_name(request),
        num_waiting=scheduler.num_waiting,
        num_running=scheduler.num_running,
    )


# --- Models ---


@router.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    """List available models."""
    model_name = _get_model_name(request)
    return ModelList(data=[ModelCard(id=model_name)])


# --- Chat Completions ---


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    chat_request: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint."""
    engine = _get_engine(request)
    model_name = _get_model_name(request)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Build prompt from messages (simple concatenation for toy model)
    prompt_parts: list[str] = []
    for msg in chat_request.messages:
        prompt_parts.append(f"{msg.role}: {msg.content}")
    prompt = "\n".join(prompt_parts) + "\nassistant: "

    sampling_params = SamplingParams(
        temperature=chat_request.temperature,
        top_p=chat_request.top_p,
        max_tokens=chat_request.max_tokens,
        stop=chat_request.get_stop_list(),
        seed=chat_request.seed,
        logit_bias=chat_request.get_logit_bias(),
    )

    if chat_request.stream:
        return StreamingResponse(
            _stream_chat(engine, prompt, sampling_params, request_id, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
            },
        )

    # Non-streaming: collect all tokens
    full_text = ""
    num_output_tokens = 0
    finish_reason: str | None = None
    num_prompt_tokens = 0

    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            full_text += output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
        num_output_tokens = output.num_output_tokens
        num_prompt_tokens = len(output.prompt_token_ids)

    return ChatCompletionResponse(
        id=request_id,
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_output_tokens,
            total_tokens=num_prompt_tokens + num_output_tokens,
        ),
    )


async def _stream_chat(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for chat completions."""
    # First chunk: role announcement
    first_chunk = ChatCompletionChunk(
        id=request_id,
        model=model_name,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    async for output in engine.generate(prompt, sampling_params, request_id):
        if output.outputs:
            chunk = ChatCompletionChunk(
                id=request_id,
                model=model_name,
                choices=[
                    ChatCompletionStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=output.outputs[0].text),
                        finish_reason=output.outputs[0].finish_reason,
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


# --- Completions ---


@router.post("/v1/completions")
async def completions(
    comp_request: CompletionRequest,
    request: Request,
) -> CompletionResponse:
    """OpenAI-compatible text completions endpoint (non-streaming)."""
    engine = _get_engine(request)
    model_name = _get_model_name(request)
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    sampling_params = SamplingParams(
        temperature=comp_request.temperature,
        top_p=comp_request.top_p,
        max_tokens=comp_request.max_tokens,
        stop=comp_request.get_stop_list(),
        seed=comp_request.seed,
    )

    full_text = ""
    num_output_tokens = 0
    finish_reason: str | None = None
    num_prompt_tokens = 0

    async for output in engine.generate(comp_request.prompt, sampling_params, request_id):
        if output.outputs:
            full_text += output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
        num_output_tokens = output.num_output_tokens
        num_prompt_tokens = len(output.prompt_token_ids)

    return CompletionResponse(
        id=request_id,
        model=model_name,
        choices=[CompletionChoice(index=0, text=full_text, finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_output_tokens,
            total_tokens=num_prompt_tokens + num_output_tokens,
        ),
    )
