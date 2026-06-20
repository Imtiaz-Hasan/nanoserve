"""Prefix cache: content-addressed hash-chaining and LRU block management."""

from __future__ import annotations

from collections import OrderedDict


def compute_block_hash(parent_hash: int | None, tokens: tuple[int, ...]) -> int:
    """Compute content hash for a physical KV block in a hash-chain.

    Formula: hash((parent_hash, tuple(tokens)))
    """
    return hash((parent_hash, tokens))


class PrefixCache:
    """Radix/content-addressed hash table indexing physical KV cache blocks.

    Enables zero-compute prefix reuse across requests sharing system prompts,
    few-shot examples, or multi-turn conversational context.
    """

    def __init__(self) -> None:
        # Map: content_hash -> physical_block_id
        self._hash_to_block: dict[int, int] = {}
        # Map: physical_block_id -> content_hash
        self._block_to_hash: dict[int, int] = {}
        # Map: physical_block_id -> token tuple
        self._block_tokens: dict[int, tuple[int, ...]] = {}
        # LRU tracker for cached physical blocks with ref_count == 0
        self._lru_blocks: OrderedDict[int, None] = OrderedDict()

    @property
    def num_cached_blocks(self) -> int:
        """Total number of physical blocks currently indexed in the prefix cache."""
        return len(self._block_to_hash)

    @property
    def num_unreferenced_blocks(self) -> int:
        """Number of cached physical blocks with 0 active references (available for eviction)."""
        return len(self._lru_blocks)

    def has_block(self, block_id: int) -> bool:
        """Whether a physical block is indexed in the prefix cache."""
        return block_id in self._block_to_hash

    def match_prefix(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> tuple[list[int], list[int]]:
        """Find the longest sequence of cached physical blocks matching the token prefix.

        Args:
            token_ids: Full prompt token sequence
            block_size: Number of tokens per physical block

        Returns:
            matched_block_ids: List of physical block IDs holding cached KV states
            matched_hashes: Corresponding content hashes in the hash chain
        """
        matched_blocks: list[int] = []
        matched_hashes: list[int] = []

        num_full_blocks = len(token_ids) // block_size
        parent_hash: int | None = None

        for i in range(num_full_blocks):
            block_tokens = tuple(token_ids[i * block_size : (i + 1) * block_size])
            current_hash = compute_block_hash(parent_hash, block_tokens)

            if current_hash in self._hash_to_block:
                block_id = self._hash_to_block[current_hash]
                matched_blocks.append(block_id)
                matched_hashes.append(current_hash)
                parent_hash = current_hash
            else:
                break

        return matched_blocks, matched_hashes

    def cache_block(
        self,
        block_id: int,
        block_hash: int,
        tokens: tuple[int, ...],
    ) -> None:
        """Index a physical block in the prefix cache."""
        # Invalidate any previous hash pointing to this block ID
        if block_id in self._block_to_hash:
            old_hash = self._block_to_hash[block_id]
            self._hash_to_block.pop(old_hash, None)

        self._hash_to_block[block_hash] = block_id
        self._block_to_hash[block_id] = block_hash
        self._block_tokens[block_id] = tokens
        self._lru_blocks.pop(block_id, None)

    def touch(self, block_id: int) -> None:
        """Mark a cached block as actively referenced (remove from LRU eviction pool)."""
        self._lru_blocks.pop(block_id, None)

    def mark_unreferenced(self, block_id: int) -> None:
        """Mark a cached block as having 0 references (eligible for LRU eviction)."""
        if block_id in self._block_to_hash:
            self._lru_blocks[block_id] = None  # Insert at tail (most recently unreferenced)

    def evict_lru(self) -> int | None:
        """Evict the least-recently-used unreferenced block from the prefix cache.

        Returns:
            Evicted physical block ID (now ready to be returned to the free list),
            or None if no unreferenced blocks are available.
        """
        if not self._lru_blocks:
            return None

        # Pop from head (least recently used)
        block_id, _ = self._lru_blocks.popitem(last=False)
        old_hash = self._block_to_hash.pop(block_id, None)
        if old_hash is not None:
            self._hash_to_block.pop(old_hash, None)
        self._block_tokens.pop(block_id, None)
        return block_id

    def invalidate_block(self, block_id: int) -> None:
        """Remove a physical block from the prefix cache (e.g. during COW or re-allocation)."""
        self._lru_blocks.pop(block_id, None)
        old_hash = self._block_to_hash.pop(block_id, None)
        if old_hash is not None:
            self._hash_to_block.pop(old_hash, None)
        self._block_tokens.pop(block_id, None)
