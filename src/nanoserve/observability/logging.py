"""Structured logging configuration with structlog."""

from __future__ import annotations

import logging

import structlog


def configure_logging(json_mode: bool = False) -> None:
    """Configure structlog for the application.

    Args:
        json_mode: if True, output JSON lines (production). If False, pretty console (dev).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_mode:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    # Also configure standard library logging to route through structlog
    logging.basicConfig(format="%(message)s", level=logging.INFO, force=True)
