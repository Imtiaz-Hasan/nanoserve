"""Block table: per-sequence logical-to-physical block mapping.

Week 1: trivial identity mapping (contiguous cache, no paging).
Week 2 replaces this with true paged block tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlockTable:
    """Maps logical block indices to physical block indices for one sequence.

    Week 1 implementation: identity mapping where logical == physical.
    The interface is designed so that Week 2's paged replacement is drop-in.
    """

    _table: list[int] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return len(self._table)

    def append_block(self, physical_block_id: int) -> None:
        """Append a new physical block mapping."""
        self._table.append(physical_block_id)

    def get_physical_block(self, logical_idx: int) -> int:
        """Return the physical block id for a logical index."""
        return self._table[logical_idx]

    def get_all_physical_blocks(self) -> list[int]:
        """Return all physical block ids in logical order."""
        return list(self._table)

    def copy(self) -> BlockTable:
        """Deep copy for fork / COW operations."""
        new = BlockTable()
        new._table = list(self._table)
        return new

    def clear(self) -> None:
        """Release all mappings."""
        self._table.clear()
