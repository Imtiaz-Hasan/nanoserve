"""Block manager: physical KV cache allocation and lifetime management.

Week 1: naive contiguous allocator — one contiguous slot range per sequence.
Week 2 replaces this with a proper paged block allocator with refcounting and COW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from nanoserve.config import CacheConfig, ModelConfig
from nanoserve.core.block_table import BlockTable
from nanoserve.core.sequence import Sequence


@dataclass
class BlockManager:
    """Manages physical KV cache block allocation.

    Week 1: each 'block' is a contiguous range of `block_size` token slots.
    Maintains a free list of block IDs. No refcounting, no COW — those are Week 2.
    """

    num_blocks: int
    block_size: int
    _free_blocks: list[int] = field(default_factory=list)
    _seq_block_tables: dict[int, BlockTable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialize free list with all block IDs (LIFO order for locality)
        self._free_blocks = list(range(self.num_blocks - 1, -1, -1))

    @classmethod
    def from_config(cls, _model: ModelConfig, cache: CacheConfig) -> BlockManager:
        """Create a block manager from configuration."""
        num_blocks = cache.num_gpu_blocks or 256
        return cls(num_blocks=num_blocks, block_size=cache.block_size)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def can_allocate(self, seq: Sequence) -> bool:
        """Check whether we have enough free blocks for this sequence."""
        needed = self._blocks_needed(seq.num_total_tokens)
        current = self._seq_block_tables.get(seq.seq_id)
        already_have = current.num_blocks if current else 0
        return (needed - already_have) <= self.num_free_blocks

    def allocate(self, seq: Sequence) -> BlockTable:
        """Allocate blocks for a sequence (or extend its existing allocation).

        Returns the updated block table.
        """
        needed = self._blocks_needed(seq.num_total_tokens)
        table = self._seq_block_tables.get(seq.seq_id)

        if table is None:
            table = BlockTable()
            self._seq_block_tables[seq.seq_id] = table

        while table.num_blocks < needed:
            if not self._free_blocks:
                msg = (
                    f"Block allocator OOM: need {needed - table.num_blocks} more blocks, "
                    f"0 free out of {self.num_blocks} total"
                )
                raise RuntimeError(msg)
            block_id = self._free_blocks.pop()
            table.append_block(block_id)

        return table

    def free(self, seq: Sequence) -> None:
        """Free all blocks held by a sequence."""
        table = self._seq_block_tables.pop(seq.seq_id, None)
        if table is not None:
            for block_id in reversed(table.get_all_physical_blocks()):
                self._free_blocks.append(block_id)
            table.clear()

    def get_block_table(self, seq: Sequence) -> BlockTable | None:
        """Return the block table for a sequence, or None if not allocated."""
        return self._seq_block_tables.get(seq.seq_id)

    def _blocks_needed(self, num_tokens: int) -> int:
        """Calculate how many blocks are needed for `num_tokens`."""
        return max(1, math.ceil(num_tokens / self.block_size))
