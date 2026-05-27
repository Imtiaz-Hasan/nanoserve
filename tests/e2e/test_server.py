"""End-to-end server tests via httpx AsyncClient (no network, ASGI transport)."""

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.server.app import create_app


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Create a test client with the toy model engine."""
    config = EngineConfig(
        model=ModelConfig(model_name_or_path="toy"),
        cache=CacheConfig(num_gpu_blocks=64, block_size=16),
        scheduler=SchedulerConfig(max_num_seqs=8),
        device="cpu",
        seed=42,
    )
    app = create_app(config)
    engine = app.state.engine
    await engine.start()

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await engine.stop()


@pytest.mark.asyncio
async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    """GET /health returns 200 with status 'ok'."""
    res = await client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["model"] == "toy"


@pytest.mark.asyncio
async def test_models_endpoint(client: httpx.AsyncClient) -> None:
    """GET /v1/models lists the toy model."""
    res = await client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "toy"


@pytest.mark.asyncio
async def test_completions_non_streaming(client: httpx.AsyncClient) -> None:
    """POST /v1/completions returns a valid non-streaming response."""
    res = await client.post(
        "/v1/completions",
        json={
            "model": "toy",
            "prompt": "Hello",
            "max_tokens": 8,
            "temperature": 0.0,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"]["completion_tokens"] == 8
    assert data["usage"]["prompt_tokens"] > 0


@pytest.mark.asyncio
async def test_chat_completions_non_streaming(client: httpx.AsyncClient) -> None:
    """POST /v1/chat/completions (non-streaming) returns a valid response."""
    res = await client.post(
        "/v1/chat/completions",
        json={
            "model": "toy",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_chat_completions_streaming(client: httpx.AsyncClient) -> None:
    """POST /v1/chat/completions (streaming) returns valid SSE chunks."""
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "toy",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        chunks: list[str] = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunks.append(line)

        # Should have: role announcement + tokens + [DONE]
        assert len(chunks) >= 3  # at least role + 1 token + DONE
        assert chunks[-1] == "data: [DONE]"
