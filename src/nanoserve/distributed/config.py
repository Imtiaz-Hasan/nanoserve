"""Configuration for distributed execution and tensor parallelism."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelConfig:
    """Configuration for multi-worker tensor parallelism."""

    tp_size: int = 1
    rank: int = 0
    backend: str = "gloo"

    @property
    def is_distributed(self) -> bool:
        """Whether tensor parallelism is enabled (> 1 worker)."""
        return self.tp_size > 1
