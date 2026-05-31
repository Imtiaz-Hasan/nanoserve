"""Block table: per-sequence logical-to-physical block mapping with Copy-On-Write support."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlockTable:
    """Maps logical block indices to physical block indices for one sequence.

    A sequence of length N with block_size B has ceil(N / B) logical blocks.
    Each logical block maps to a physical block ID managed by the BlockManager.
    """

    _table: list[int] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        """Total number of mapped physical blocks."""
        return len(self._table)

    def append_block(self, physical_block_id: int) -> None:
        """Append a newly allocated physical block mapping."""
        self._table.append(physical_block_id)

    def get_physical_block(self, logical_idx: int) -> int:
        """Return the physical block id for a logical block index."""
        if logical_idx < 0 or logical_idx >= len(self._table):
            msg = f"Logical index {logical_idx} out of range (0..{len(self._table) - 1})"
            raise IndexError(msg)
        return self._table[logical_idx]

    def set_physical_block(self, logical_idx: int, physical_block_id: int) -> None:
        """Update mapping at logical_idx (used during Copy-On-Write)."""
        if logical_idx < 0 or logical_idx >= len(self._table):
            msg = f"Logical index {logical_idx} out of range (0..{len(self._table) - 1})"
            raise IndexError(msg)
        self._table[logical_idx] = physical_block_id

    def get_all_physical_blocks(self) -> list[int]:
        """Return all physical block IDs in logical sequence order."""
        return list(self._table)

    def copy(self) -> BlockTable:
        """Create a shallow copy of the block table (used for sequence forking)."""
        new_table = BlockTable()
        new_table._table = list(self._table)
        return new_table

    def clear(self) -> None:
        """Release all block mappings."""
        self._table.clear()

    def __getitem__(self, idx: int) -> int:
        return self._table[idx]

    def __len__(self) -> int:
        return len(self._table)
