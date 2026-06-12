"""Block manager: physical KV cache allocation, refcounting, Copy-On-Write (COW), and CPU swap."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from nanoserve.config import CacheConfig, ModelConfig
from nanoserve.core.block_table import BlockTable
from nanoserve.core.sequence import Sequence


@dataclass
class PhysicalBlock:
    """Descriptor for a single physical block in the KV cache."""

    block_id: int
    ref_count: int = 0
    device: str = "gpu"

    @property
    def is_free(self) -> bool:
        """A block is free if its reference count is 0."""
        return self.ref_count == 0


@dataclass
class BlockManager:
    """Manages physical KV cache block allocation, refcounts, COW, and CPU swap pools.

    Maintains dual pools:
      - GPU physical block pool for active decoding/prefilling
      - CPU physical block pool for swapped/preempted sequences
    """

    num_blocks: int
    block_size: int
    num_cpu_blocks: int = 256
    _blocks: list[PhysicalBlock] = field(default_factory=list)
    _free_blocks: list[int] = field(default_factory=list)
    _cpu_blocks: list[PhysicalBlock] = field(default_factory=list)
    _cpu_free_blocks: list[int] = field(default_factory=list)
    _seq_block_tables: dict[int, BlockTable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialize GPU physical blocks and free list (LIFO order)
        self._blocks = [
            PhysicalBlock(block_id=i, ref_count=0, device="gpu") for i in range(self.num_blocks)
        ]
        self._free_blocks = list(range(self.num_blocks - 1, -1, -1))

        # Initialize CPU physical blocks and free list
        self._cpu_blocks = [
            PhysicalBlock(block_id=i, ref_count=0, device="cpu") for i in range(self.num_cpu_blocks)
        ]
        self._cpu_free_blocks = list(range(self.num_cpu_blocks - 1, -1, -1))

    @classmethod
    def from_config(cls, _model: ModelConfig, cache: CacheConfig) -> BlockManager:
        """Create a block manager from configuration."""
        num_blocks = cache.num_gpu_blocks or 256
        return cls(
            num_blocks=num_blocks,
            block_size=cache.block_size,
            num_cpu_blocks=cache.num_cpu_blocks,
        )

    @property
    def num_free_blocks(self) -> int:
        """Number of unallocated GPU physical blocks."""
        return len(self._free_blocks)

    @property
    def num_used_blocks(self) -> int:
        """Number of GPU physical blocks currently referenced by at least one sequence."""
        return self.num_blocks - self.num_free_blocks

    @property
    def num_cpu_free_blocks(self) -> int:
        """Number of unallocated CPU physical blocks."""
        return len(self._cpu_free_blocks)

    @property
    def num_cpu_used_blocks(self) -> int:
        """Number of CPU physical blocks currently referenced by at least one sequence."""
        return self.num_cpu_blocks - self.num_cpu_free_blocks

    @property
    def total_ref_count(self) -> int:
        """Sum of ref counts across all GPU physical blocks."""
        return sum(b.ref_count for b in self._blocks)

    def get_block_ref_count(self, block_id: int) -> int:
        """Get reference count of a specific GPU physical block."""
        return self._blocks[block_id].ref_count

    def can_allocate(self, seq: Sequence) -> bool:
        """Check whether we have enough free GPU blocks to allocate/extend this sequence."""
        needed = self._blocks_needed(seq.num_total_tokens)
        current = self._seq_block_tables.get(seq.seq_id)
        already_have = current.num_blocks if current else 0
        return (needed - already_have) <= self.num_free_blocks

    def allocate(self, seq: Sequence) -> BlockTable:
        """Allocate GPU blocks for a sequence (or extend its existing allocation)."""
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
            self._blocks[block_id].ref_count = 1
            table.append_block(block_id)

        return table

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> BlockTable:
        """Fork a child sequence from a parent sequence with zero-copy block sharing."""
        parent_table = self._seq_block_tables.get(parent_seq.seq_id)
        if parent_table is None:
            msg = f"Parent sequence {parent_seq.seq_id} has no allocated block table"
            raise ValueError(msg)

        child_table = parent_table.copy()
        for block_id in child_table.get_all_physical_blocks():
            self._blocks[block_id].ref_count += 1

        self._seq_block_tables[child_seq.seq_id] = child_table
        return child_table

    def cow(self, seq: Sequence, logical_block_idx: int) -> tuple[int, int] | None:
        """Perform Copy-On-Write if the logical block is shared (ref_count > 1)."""
        table = self._seq_block_tables.get(seq.seq_id)
        if table is None:
            msg = f"Sequence {seq.seq_id} has no allocated block table"
            raise ValueError(msg)

        old_block_id = table.get_physical_block(logical_block_idx)
        if self._blocks[old_block_id].ref_count <= 1:
            return None

        if not self._free_blocks:
            msg = "Block allocator OOM during Copy-On-Write"
            raise RuntimeError(msg)

        new_block_id = self._free_blocks.pop()
        self._blocks[new_block_id].ref_count = 1
        self._blocks[old_block_id].ref_count -= 1
        table.set_physical_block(logical_block_idx, new_block_id)
        return old_block_id, new_block_id

    def can_swap_out(self, seq: Sequence) -> bool:
        """Check if CPU memory pool has enough free blocks to swap out sequence."""
        table = self._seq_block_tables.get(seq.seq_id)
        if table is None:
            return True
        return len(self._cpu_free_blocks) >= table.num_blocks

    def swap_out(self, seq: Sequence) -> dict[int, int]:
        """Swap sequence physical blocks from GPU memory pool to CPU memory pool.

        Returns:
            Mapping from GPU physical block ID to assigned CPU physical block ID.
        """
        table = self._seq_block_tables.get(seq.seq_id)
        if table is None:
            return {}

        mapping: dict[int, int] = {}
        for logical_idx, gpu_block_id in enumerate(table.get_all_physical_blocks()):
            if not self._cpu_free_blocks:
                msg = "CPU Block allocator OOM during swap_out"
                raise RuntimeError(msg)
            cpu_block_id = self._cpu_free_blocks.pop()
            self._cpu_blocks[cpu_block_id].ref_count = 1

            # Release GPU block
            self._blocks[gpu_block_id].ref_count -= 1
            if self._blocks[gpu_block_id].ref_count == 0:
                self._free_blocks.append(gpu_block_id)

            table.set_physical_block(logical_idx, cpu_block_id)
            mapping[gpu_block_id] = cpu_block_id

        return mapping

    def can_swap_in(self, seq: Sequence) -> bool:
        """Check if GPU memory pool has enough free blocks to swap in sequence."""
        table = self._seq_block_tables.get(seq.seq_id)
        if table is None:
            return True
        return len(self._free_blocks) >= table.num_blocks

    def swap_in(self, seq: Sequence) -> dict[int, int]:
        """Swap sequence physical blocks from CPU memory pool to GPU memory pool.

        Returns:
            Mapping from CPU physical block ID to assigned GPU physical block ID.
        """
        table = self._seq_block_tables.get(seq.seq_id)
        if table is None:
            return {}

        mapping: dict[int, int] = {}
        for logical_idx, cpu_block_id in enumerate(table.get_all_physical_blocks()):
            if not self._free_blocks:
                msg = "GPU Block allocator OOM during swap_in"
                raise RuntimeError(msg)
            gpu_block_id = self._free_blocks.pop()
            self._blocks[gpu_block_id].ref_count = 1

            # Release CPU block
            self._cpu_blocks[cpu_block_id].ref_count -= 1
            if self._cpu_blocks[cpu_block_id].ref_count == 0:
                self._cpu_free_blocks.append(cpu_block_id)

            table.set_physical_block(logical_idx, gpu_block_id)
            mapping[cpu_block_id] = gpu_block_id

        return mapping

    def free(self, seq: Sequence) -> None:
        """Free all blocks held by a sequence (from GPU or CPU pools)."""
        table = self._seq_block_tables.pop(seq.seq_id, None)
        if table is not None:
            for block_id in reversed(table.get_all_physical_blocks()):
                if block_id < len(self._blocks) and self._blocks[block_id].ref_count > 0:
                    self._blocks[block_id].ref_count -= 1
                    if self._blocks[block_id].ref_count == 0:
                        self._free_blocks.append(block_id)
                elif block_id < len(self._cpu_blocks) and self._cpu_blocks[block_id].ref_count > 0:
                    self._cpu_blocks[block_id].ref_count -= 1
                    if self._cpu_blocks[block_id].ref_count == 0:
                        self._cpu_free_blocks.append(block_id)
            table.clear()

    def get_block_table(self, seq: Sequence) -> BlockTable | None:
        """Return the block table for a sequence, or None if not allocated."""
        return self._seq_block_tables.get(seq.seq_id)

    def get_slot_mapping(self, seq: Sequence, positions: list[int]) -> list[int]:
        """Compute flat physical slot index for token positions."""
        table = self._seq_block_tables.get(seq.seq_id)
        if table is None:
            msg = f"Sequence {seq.seq_id} has no allocated block table"
            raise ValueError(msg)

        slots: list[int] = []
        for pos in positions:
            logical_block_idx = pos // self.block_size
            offset = pos % self.block_size
            physical_block_id = table.get_physical_block(logical_block_idx)
            slots.append(physical_block_id * self.block_size + offset)
        return slots

    def _blocks_needed(self, num_tokens: int) -> int:
        """Calculate how many blocks are needed for `num_tokens`."""
        return max(1, math.ceil(num_tokens / self.block_size))
