"""AsyncLLMEngine: asyncio wrapper around the synchronous LLMEngine.

Provides per-request async generators for streaming tokens.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncGenerator

from nanoserve.config import EngineConfig
from nanoserve.engine.llm_engine import LLMEngine
from nanoserve.engine.output import RequestOutput
from nanoserve.sampling.params import SamplingParams

logger = logging.getLogger(__name__)


class AsyncLLMEngine:
    """Async wrapper that runs the LLMEngine step loop in a background task.

    Each call to generate() returns an AsyncGenerator that yields RequestOutputs
    as tokens are generated. Multiple concurrent requests are supported.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._engine: LLMEngine | None = None
        self._request_queues: dict[str, asyncio.Queue[RequestOutput | None]] = {}
        self._loop_task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Initialize the engine and start the background step loop."""
        logger.info("Starting async engine...")
        self._engine = LLMEngine(self.config)
        self._running = True
        self._loop_task = asyncio.create_task(self._engine_loop())
        logger.info("Async engine started.")

    async def stop(self) -> None:
        """Stop the background loop and clean up."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        # Signal all waiting generators
        for q in self._request_queues.values():
            await q.put(None)
        self._request_queues.clear()
        logger.info("Async engine stopped.")

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Submit a request and yield outputs as they are generated.

        Args:
            prompt: input text
            sampling_params: sampling configuration
            request_id: optional request ID (auto-generated if None)

        Yields:
            RequestOutput for each generated token, with finished=True on the last one.
        """
        if self._engine is None:
            msg = "Engine not started. Call start() first."
            raise RuntimeError(msg)

        if request_id is None:
            request_id = f"req-{uuid.uuid4().hex[:12]}"

        queue: asyncio.Queue[RequestOutput | None] = asyncio.Queue()
        self._request_queues[request_id] = queue

        self._engine.add_request(request_id, prompt, sampling_params)

        try:
            while True:
                output = await queue.get()
                if output is None:
                    break
                yield output
                if output.finished:
                    break
        finally:
            self._request_queues.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        """Abort a running request."""
        if self._engine is not None:
            self._engine.abort_request(request_id)
        queue = self._request_queues.pop(request_id, None)
        if queue is not None:
            await queue.put(None)

    async def _engine_loop(self) -> None:
        """Background loop: repeatedly call step() and dispatch results."""
        while self._running:
            if self._engine is None:
                break

            if not self._engine.has_unfinished_requests():
                # No work to do — yield control briefly
                await asyncio.sleep(0.01)
                continue

            # Run the synchronous step in the default executor to avoid blocking
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(None, self._engine.step)

            for output in outputs:
                queue = self._request_queues.get(output.request_id)
                if queue is not None:
                    await queue.put(output)

            # Small yield to allow other coroutines to run
            await asyncio.sleep(0)

    @property
    def engine(self) -> LLMEngine:
        """Access the underlying synchronous engine."""
        if self._engine is None:
            msg = "Engine not started."
            raise RuntimeError(msg)
        return self._engine
