"""FastAPI application factory and CLI entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from nanoserve.config import CacheConfig, EngineConfig, ModelConfig, SchedulerConfig
from nanoserve.engine.async_engine import AsyncLLMEngine
from nanoserve.server.middleware import RequestIdMiddleware
from nanoserve.server.openai_routes import router

logger = structlog.get_logger(__name__)


def create_app(engine_config: EngineConfig) -> FastAPI:
    """Create the FastAPI application with lifespan management."""
    engine = AsyncLLMEngine(engine_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Starting nanoserve engine...", model=engine_config.model.model_name_or_path)
        await engine.start()
        app.state.engine = engine
        app.state.model_name = engine_config.model.model_name_or_path
        logger.info("nanoserve ready.", device=engine_config.device)
        yield
        logger.info("Shutting down nanoserve...")
        await engine.stop()

    app = FastAPI(
        title="nanoserve",
        description="Production-shaped LLM serving engine with paged KV cache",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.model_name = engine_config.model.model_name_or_path
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)

    return app


def main() -> None:
    """CLI entrypoint: parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        prog="nanoserve",
        description="nanoserve: Production-shaped LLM serving engine",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", type=str, default="toy", help="Model name or path")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--num-blocks", type=int, default=256, help="Number of KV cache blocks")
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    parser.add_argument("--max-num-seqs", type=int, default=128, help="Max concurrent sequences")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    model_config = ModelConfig(model_name_or_path=args.model)
    cache_config = CacheConfig(
        block_size=args.block_size,
        num_gpu_blocks=args.num_blocks,
    )
    scheduler_config = SchedulerConfig(max_num_seqs=args.max_num_seqs)

    engine_config = EngineConfig(
        model=model_config,
        cache=cache_config,
        scheduler=scheduler_config,
        device=args.device,
        seed=args.seed,
    )

    app = create_app(engine_config)

    print(
        f"[NANOSERVE] Starting on http://{args.host}:{args.port}\n"
        f"  Model: {args.model}\n"
        f"  Device: {args.device}\n"
        f"  Blocks: {args.num_blocks} × {args.block_size} tokens\n"
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
