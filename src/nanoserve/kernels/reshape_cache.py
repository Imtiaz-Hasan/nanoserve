"""PagedAttention cache kernel utilities: scattering, gathering, and block copying."""

from __future__ import annotations

import torch


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter computed key and value tokens into pre-allocated physical KV cache slots.

    Args:
        key: (num_tokens, num_kv_heads, head_dim) or (batch, num_kv_heads, num_tokens, head_dim)
        value: (num_tokens, num_kv_heads, head_dim) or (batch, num_kv_heads, num_tokens, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) flat slot indices
    """
    if key.dim() == 4:
        # (batch, num_kv_heads, num_tokens, head_dim) -> (num_tokens, num_kv_heads, head_dim)
        key = key.squeeze(0).transpose(0, 1)
        value = value.squeeze(0).transpose(0, 1)

    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]

    # Flatten cache along physical block and block_size dimensions
    k_flat = k_cache.view(-1, num_kv_heads, head_dim)
    v_flat = v_cache.view(-1, num_kv_heads, head_dim)

    # In-place scatter write
    k_flat[slot_mapping] = key.to(dtype=k_flat.dtype)
    v_flat[slot_mapping] = value.to(dtype=v_flat.dtype)


def gather_paged_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: list[int],
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather scattered physical blocks into a contiguous tensor for reference SDPA attention.

    Args:
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_table: list of physical block IDs assigned to this sequence
        num_tokens: total sequence length

    Returns:
        gathered_k: (1, num_kv_heads, num_tokens, head_dim)
        gathered_v: (1, num_kv_heads, num_tokens, head_dim)
    """
    if not block_table:
        num_kv_heads = k_cache.shape[2]
        head_dim = k_cache.shape[3]
        device = k_cache.device
        dtype = k_cache.dtype
        empty = torch.empty((1, num_kv_heads, 0, head_dim), dtype=dtype, device=device)
        return empty, empty

    block_indices = torch.tensor(block_table, dtype=torch.long, device=k_cache.device)

    # Index physical blocks from cache
    k_selected = k_cache[block_indices]
    v_selected = v_cache[block_indices]

    num_seq_blocks, block_size, num_kv_heads, head_dim = k_selected.shape

    # Reshape to (num_seq_blocks * block_size, num_kv_heads, head_dim)
    k_flat = k_selected.view(num_seq_blocks * block_size, num_kv_heads, head_dim)
    v_flat = v_selected.view(num_seq_blocks * block_size, num_kv_heads, head_dim)

    # Truncate to exact active sequence length
    k_active = k_flat[:num_tokens]  # (num_tokens, num_kv_heads, head_dim)
    v_active = v_flat[:num_tokens]

    # Transpose to (1, num_kv_heads, num_tokens, head_dim)
    gathered_k = k_active.unsqueeze(0).transpose(1, 2)
    gathered_v = v_active.unsqueeze(0).transpose(1, 2)

    return gathered_k, gathered_v


def copy_block_data(
    src_block_id: int,
    dst_block_id: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> None:
    """Copy physical KV tensor activations from source block to destination block (during COW)."""
    k_cache[dst_block_id].copy_(k_cache[src_block_id])
    v_cache[dst_block_id].copy_(v_cache[src_block_id])
