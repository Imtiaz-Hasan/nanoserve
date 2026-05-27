"""Placeholder for content-addressed prefix cache. Implemented in Week 7."""

from __future__ import annotations


class PrefixCache:
    """Hash-chained content-addressed prefix cache with refcounting and LRU eviction.

    Week 1: stub. Week 7 implements the full cache.
    """

    def __init__(self, block_size: int = 16) -> None:
        self.block_size = block_size
        self._cache: dict[int, int] = {}

    @property
    def hit_rate(self) -> float:
        """Return the current cache hit rate. Stub returns 0.0."""
        return 0.0
