"""Unit tests for block manager: allocation, free, OOM, budget enforcement."""

import pytest

from nanoserve.core.block_manager import BlockManager
from nanoserve.core.sequence import Sequence


def test_allocate_and_free() -> None:
    """Basic allocate → use → free lifecycle."""
    bm = BlockManager(num_blocks=10, block_size=16)
    assert bm.num_free_blocks == 10

    seq = Sequence(seq_id=0, prompt_token_ids=list(range(32)))  # 32 tokens → 2 blocks
    table = bm.allocate(seq)

    assert table.num_blocks == 2
    assert bm.num_free_blocks == 8

    bm.free(seq)
    assert bm.num_free_blocks == 10


def test_can_allocate() -> None:
    """can_allocate correctly predicts whether allocation will succeed."""
    bm = BlockManager(num_blocks=3, block_size=16)

    seq_small = Sequence(seq_id=0, prompt_token_ids=list(range(16)))  # 1 block
    seq_big = Sequence(seq_id=1, prompt_token_ids=list(range(64)))  # 4 blocks

    assert bm.can_allocate(seq_small)
    assert not bm.can_allocate(seq_big)


def test_oom_raises() -> None:
    """Allocating more blocks than available raises RuntimeError."""
    bm = BlockManager(num_blocks=2, block_size=16)
    seq = Sequence(seq_id=0, prompt_token_ids=list(range(64)))  # needs 4 blocks

    with pytest.raises(RuntimeError, match="OOM"):
        bm.allocate(seq)


def test_extend_allocation() -> None:
    """Allocating for a sequence that grows extends its block table."""
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = Sequence(seq_id=0, prompt_token_ids=list(range(16)))  # 1 block initially

    table = bm.allocate(seq)
    assert table.num_blocks == 1
    assert bm.num_free_blocks == 9

    # Simulate generating 16 more tokens
    for i in range(16):
        seq.append_token(100 + i)

    table = bm.allocate(seq)
    assert table.num_blocks == 2
    assert bm.num_free_blocks == 8


def test_free_idempotent() -> None:
    """Freeing a non-allocated sequence is a no-op."""
    bm = BlockManager(num_blocks=10, block_size=16)
    seq = Sequence(seq_id=99, prompt_token_ids=[1])
    bm.free(seq)  # should not raise
    assert bm.num_free_blocks == 10


def test_multiple_sequences() -> None:
    """Multiple sequences can be allocated concurrently."""
    bm = BlockManager(num_blocks=10, block_size=16)

    seqs = [Sequence(seq_id=i, prompt_token_ids=list(range(16))) for i in range(5)]
    for seq in seqs:
        bm.allocate(seq)

    assert bm.num_free_blocks == 5
    assert bm.num_used_blocks == 5

    for seq in seqs:
        bm.free(seq)

    assert bm.num_free_blocks == 10
