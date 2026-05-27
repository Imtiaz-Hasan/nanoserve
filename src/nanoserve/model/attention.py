"""Attention dispatch: reference SDPA implementation for Week 1.

Week 2 adds paged gather. Week 8 adds the Triton kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference attention using PyTorch's scaled_dot_product_attention.

    Args:
        query: (batch, num_heads, q_len, head_dim)
        key: (batch, num_kv_heads, kv_len, head_dim)
        value: (batch, num_kv_heads, kv_len, head_dim)
        scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output: (batch, num_heads, q_len, head_dim)
    """
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]

    # GQA: expand KV heads to match query heads
    if num_kv_heads < num_heads:
        repeat_factor = num_heads // num_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        is_causal=True,
        scale=scale,
    )
