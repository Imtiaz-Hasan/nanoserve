"""Correctness and numerical parity tests for paged attention decode kernel."""

import math
import random

import pytest
import torch

from nanoserve.kernels.paged_attention import paged_attention_decode
from nanoserve.kernels.reshape_cache import gather_paged_kv
from nanoserve.model.attention import reference_attention


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("block_size", [4, 16])
def test_paged_attention_decode_parity_with_reference(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
) -> None:
    """Verify paged_attention_decode matches gather_paged_kv + reference within tolerance."""
    torch.manual_seed(42)
    random.seed(42)

    max_blocks = 32
    num_kv_heads = num_heads

    # Random physical KV caches: (num_blocks, block_size, num_kv_heads, head_dim)
    k_cache = torch.randn(max_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float32)
    v_cache = torch.randn(max_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float32)

    # Random query vectors for decode step: (B, H, 1, D)
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)

    # Random sequence lengths and block tables
    seq_lens: list[int] = []
    block_tables: list[list[int]] = []

    for _ in range(batch_size):
        seq_len = random.randint(1, 48)
        seq_lens.append(seq_len)
        num_needed_blocks = math.ceil(seq_len / block_size)
        blocks = random.sample(range(max_blocks), num_needed_blocks)
        block_tables.append(blocks)

    scale = 1.0 / math.sqrt(head_dim)

    # 1. Compute via paged_attention_decode
    actual_out = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
    )

    # 2. Compute via reference gather-and-attention
    expected_outputs: list[torch.Tensor] = []
    for b in range(batch_size):
        gathered_k, gathered_v = gather_paged_kv(k_cache, v_cache, block_tables[b], seq_lens[b])
        out_b = reference_attention(q[b : b + 1], gathered_k, gathered_v, scale=scale)
        expected_outputs.append(out_b)
    expected_out = torch.cat(expected_outputs, dim=0)

    # Numerical parity check
    torch.testing.assert_close(actual_out, expected_out, atol=1e-4, rtol=1e-4)


def test_paged_attention_decode_gqa_expansion() -> None:
    """Verify paged attention correctly handles Grouped-Query Attention (GQA)."""
    batch_size = 2
    num_heads = 8
    num_kv_heads = 2  # 4 query heads per KV head
    head_dim = 64
    block_size = 4
    max_blocks = 16

    k_cache = torch.randn(max_blocks, block_size, num_kv_heads, head_dim)
    v_cache = torch.randn(max_blocks, block_size, num_kv_heads, head_dim)
    q = torch.randn(batch_size, num_heads, 1, head_dim)

    block_tables = [[0, 1, 2], [3, 4]]
    seq_lens = [10, 7]

    out = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )

    assert out.shape == (batch_size, num_heads, 1, head_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_paged_attention_zero_length_safety() -> None:
    """Verify kernel safely handles zero length sequences without crashing."""
    k_cache = torch.randn(4, 4, 2, 32)
    v_cache = torch.randn(4, 4, 2, 32)
    q = torch.randn(1, 2, 1, 32)

    out = paged_attention_decode(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=[[]],
        seq_lens=[0],
    )
    assert out.shape == (1, 2, 1, 32)
    assert (out == 0.0).all()
