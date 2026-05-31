"""Hypothesis property tests for BlockManager memory invariants and leak-freedom."""

import random

from hypothesis import given, settings
from hypothesis import strategies as st

from nanoserve.core.block_manager import BlockManager
from nanoserve.core.sequence import Sequence


@settings(max_examples=50, deadline=None)
@given(
    num_blocks=st.integers(min_value=32, max_value=256),
    block_size=st.sampled_from([4, 8, 16]),
    operations=st.lists(
        st.tuples(
            st.sampled_from(["allocate", "fork", "cow", "free"]),
            st.integers(min_value=1, max_value=64),  # sequence token length or index
        ),
        min_size=20,
        max_size=200,
    ),
)
def test_block_manager_invariants_fuzz(
    num_blocks: int,
    block_size: int,
    operations: list[tuple[str, int]],
) -> None:
    """Fuzz test block manager over randomized allocate/fork/cow/free sequences.

    Enforces 5 core memory invariants throughout the lifecycle.
    """
    bm = BlockManager(num_blocks=num_blocks, block_size=block_size)
    active_seqs: dict[int, Sequence] = {}
    next_seq_id = 1

    for op, arg in operations:
        if op == "allocate":
            token_count = arg
            seq = Sequence(seq_id=next_seq_id, prompt_token_ids=list(range(token_count)))
            if bm.can_allocate(seq):
                bm.allocate(seq)
                active_seqs[next_seq_id] = seq
                next_seq_id += 1

        elif op == "fork" and active_seqs:
            parent_id = random.choice(list(active_seqs.keys()))
            parent_seq = active_seqs[parent_id]
            child_seq = parent_seq.fork(new_seq_id=next_seq_id)
            bm.fork(parent_seq, child_seq)
            active_seqs[next_seq_id] = child_seq
            next_seq_id += 1

        elif op == "cow" and active_seqs:
            seq_id = random.choice(list(active_seqs.keys()))
            seq = active_seqs[seq_id]
            table = bm.get_block_table(seq)
            if table and table.num_blocks > 0:
                block_idx = arg % table.num_blocks
                if bm.num_free_blocks > 0 or bm.get_block_ref_count(table[block_idx]) <= 1:
                    bm.cow(seq, block_idx)

        elif op == "free" and active_seqs:
            seq_id = random.choice(list(active_seqs.keys()))
            seq = active_seqs.pop(seq_id)
            bm.free(seq)

        # Invariant 1: Total blocks conservation
        assert bm.num_free_blocks + bm.num_used_blocks == num_blocks

        # Invariant 2: Free list elements are distinct and have ref_count == 0
        assert len(set(bm._free_blocks)) == len(bm._free_blocks)
        for b_id in bm._free_blocks:
            assert bm.get_block_ref_count(b_id) == 0

        # Invariant 3: No physical block has negative refcount
        for b_id in range(num_blocks):
            assert bm.get_block_ref_count(b_id) >= 0

        # Invariant 4: Total refcounts equals sum of mapped blocks across all active tables
        expected_total_refs = sum(
            bm.get_block_table(s).num_blocks
            for s in active_seqs.values()
            if bm.get_block_table(s) is not None
        )
        assert bm.total_ref_count == expected_total_refs

    # Free all remaining sequences
    for seq in list(active_seqs.values()):
        bm.free(seq)
    active_seqs.clear()

    # Invariant 5: Zero leaks — all blocks return to free list after all sequences freed
    assert bm.num_free_blocks == num_blocks
    assert bm.num_used_blocks == 0
    assert bm.total_ref_count == 0
