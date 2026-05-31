"""Unit tests for PagedAttention block manager, refcounting, COW, and kernel utilities."""

import torch

from nanoserve.core.block_manager import BlockManager
from nanoserve.core.sequence import Sequence
from nanoserve.kernels.reshape_cache import copy_block_data, gather_paged_kv, reshape_and_cache


def test_ref_counting_and_fork() -> None:
    """Verify ref counts increment on fork and decrement on free."""
    bm = BlockManager(num_blocks=16, block_size=16)

    parent = Sequence(seq_id=1, prompt_token_ids=list(range(32)))  # 2 blocks
    table = bm.allocate(parent)
    b0, b1 = table.get_all_physical_blocks()

    assert bm.num_free_blocks == 14
    assert bm.num_used_blocks == 2
    assert bm.get_block_ref_count(b0) == 1
    assert bm.get_block_ref_count(b1) == 1

    # Fork child sequence
    child = parent.fork(new_seq_id=2)
    bm.fork(parent, child)

    # Blocks are shared; free blocks count doesn't change
    assert bm.num_free_blocks == 14
    assert bm.get_block_ref_count(b0) == 2
    assert bm.get_block_ref_count(b1) == 2

    # Free parent: blocks still held by child, refcount becomes 1
    bm.free(parent)
    assert bm.num_free_blocks == 14
    assert bm.get_block_ref_count(b0) == 1
    assert bm.get_block_ref_count(b1) == 1

    # Free child: blocks returned to free list, refcount becomes 0
    bm.free(child)
    assert bm.num_free_blocks == 16
    assert bm.get_block_ref_count(b0) == 0
    assert bm.get_block_ref_count(b1) == 0


def test_copy_on_write_isolation() -> None:
    """Verify COW allocates private block and decouples shared blocks."""
    bm = BlockManager(num_blocks=16, block_size=16)

    parent = Sequence(seq_id=1, prompt_token_ids=list(range(16)))  # 1 block
    parent_table = bm.allocate(parent)
    b0 = parent_table.get_physical_block(0)

    child = parent.fork(new_seq_id=2)
    child_table = bm.fork(parent, child)

    assert bm.get_block_ref_count(b0) == 2

    # Child writes to logical block 0 -> triggers COW
    cow_res = bm.cow(child, logical_block_idx=0)
    assert cow_res is not None
    old_block, new_block = cow_res
    assert old_block == b0
    assert new_block == child_table.get_physical_block(0)
    assert new_block != old_block

    # Parent still holds b0 with refcount 1
    assert bm.get_block_ref_count(b0) == 1
    # Child holds new_block with refcount 1
    assert bm.get_block_ref_count(new_block) == 1
    assert bm.num_free_blocks == 14

    # Subsequent COW on child is a no-op (exclusive)
    assert bm.cow(child, logical_block_idx=0) is None


def test_slot_mapping_computation() -> None:
    """Verify flat physical slot indices across multiple non-contiguous blocks."""
    bm = BlockManager(num_blocks=16, block_size=4)

    seq = Sequence(seq_id=1, prompt_token_ids=list(range(8)))  # 2 blocks
    table = bm.allocate(seq)
    p0, p1 = table.get_all_physical_blocks()  # e.g., 15 and 14

    # Positions 0..7
    slots = bm.get_slot_mapping(seq, list(range(8)))
    expected = [
        p0 * 4 + 0,
        p0 * 4 + 1,
        p0 * 4 + 2,
        p0 * 4 + 3,
        p1 * 4 + 0,
        p1 * 4 + 1,
        p1 * 4 + 2,
        p1 * 4 + 3,
    ]
    assert slots == expected


def test_reshape_and_cache_and_gather_roundtrip() -> None:
    """Verify scattering into physical paged cache and gathering matches original tensors."""
    num_blocks = 8
    block_size = 4
    num_kv_heads = 2
    head_dim = 8
    seq_len = 7  # Spans 2 blocks (4 tokens in block 0, 3 in block 1)

    k_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=torch.float32)
    v_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype=torch.float32)

    # Simulated K, V projections for 7 tokens: (seq_len, num_kv_heads, head_dim)
    torch.manual_seed(42)
    k_src = torch.randn(seq_len, num_kv_heads, head_dim)
    v_src = torch.randn(seq_len, num_kv_heads, head_dim)

    # Assign non-contiguous physical blocks: block 5 and block 2
    block_table = [5, 2]
    slots = [
        5 * 4 + 0,
        5 * 4 + 1,
        5 * 4 + 2,
        5 * 4 + 3,
        2 * 4 + 0,
        2 * 4 + 1,
        2 * 4 + 2,
    ]
    slot_mapping = torch.tensor(slots, dtype=torch.long)

    # 1. Scatter write
    reshape_and_cache(k_src, v_src, k_cache, v_cache, slot_mapping)

    # 2. Gather read
    gathered_k, gathered_v = gather_paged_kv(k_cache, v_cache, block_table, num_tokens=seq_len)

    # gathered shape: (1, num_kv_heads, seq_len, head_dim)
    expected_k = k_src.unsqueeze(0).transpose(1, 2)
    expected_v = v_src.unsqueeze(0).transpose(1, 2)

    torch.testing.assert_close(gathered_k, expected_k)
    torch.testing.assert_close(gathered_v, expected_v)


def test_copy_block_data() -> None:
    """Verify physical block data copy."""
    k_cache = torch.zeros((4, 4, 2, 4))
    v_cache = torch.zeros((4, 4, 2, 4))

    k_cache[1] = torch.ones((4, 2, 4)) * 3.14
    v_cache[1] = torch.ones((4, 2, 4)) * 2.71

    copy_block_data(src_block_id=1, dst_block_id=3, k_cache=k_cache, v_cache=v_cache)

    torch.testing.assert_close(k_cache[3], k_cache[1])
    torch.testing.assert_close(v_cache[3], v_cache[1])
