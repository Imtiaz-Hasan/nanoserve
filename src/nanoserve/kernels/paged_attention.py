"""Paged attention decode kernel: fused Triton GPU kernel with PyTorch CPU fallback."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:

    @triton.jit  # type: ignore[untyped-decorator]
    def _paged_attention_decode_kernel(
        out_ptr: Any,
        q_ptr: Any,
        k_cache_ptr: Any,
        v_cache_ptr: Any,
        block_tables_ptr: Any,
        seq_lens_ptr: Any,
        scale: float,
        stride_qb: int,
        stride_qh: int,
        stride_qd: int,
        stride_kb: int,
        stride_ks: int,
        stride_kh: int,
        stride_kd: int,
        stride_vb: int,
        stride_vs: int,
        stride_vh: int,
        stride_vd: int,
        stride_ob: int,
        stride_oh: int,
        stride_od: int,
        stride_bt_b: int,
        stride_bt_idx: int,
        BLOCK_SIZE: Any,  # noqa: N803
        HEAD_DIM: Any,  # noqa: N803
    ) -> None:
        """Fused Triton decode attention kernel with online softmax."""
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        seq_len = tl.load(seq_lens_ptr + batch_idx)
        if seq_len <= 0:
            return

        # Load Query vector (1, HEAD_DIM)
        offs_d = tl.arange(0, HEAD_DIM)
        q_ptrs = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd
        q = tl.load(q_ptrs)

        # Running online softmax state
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        for block_idx in range(num_blocks):
            # Load physical block ID from block table
            bt_ptr = block_tables_ptr + batch_idx * stride_bt_b + block_idx * stride_bt_idx
            physical_block_id = tl.load(bt_ptr)

            offs_s = tl.arange(0, BLOCK_SIZE)
            token_indices = block_idx * BLOCK_SIZE + offs_s
            mask_s = token_indices < seq_len

            # Load K block: (BLOCK_SIZE, HEAD_DIM)
            k_ptrs = (
                k_cache_ptr
                + physical_block_id * stride_kb
                + offs_s[:, None] * stride_ks
                + head_idx * stride_kh
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask_s[:, None], other=0.0)

            # Compute QK^T * scale
            qk = tl.sum(q[None, :] * k, axis=1) * scale
            qk = tl.where(mask_s, qk, -float("inf"))

            # Online softmax update
            block_max = tl.max(qk, axis=0)
            new_m_i = tl.maximum(m_i, block_max)
            alpha = tl.exp(m_i - new_m_i)
            p = tl.exp(qk - new_m_i)

            l_i = l_i * alpha + tl.sum(p, axis=0)

            # Load V block: (BLOCK_SIZE, HEAD_DIM)
            v_ptrs = (
                v_cache_ptr
                + physical_block_id * stride_vb
                + offs_s[:, None] * stride_vs
                + head_idx * stride_vh
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask_s[:, None], other=0.0)

            # Accumulate weighted V
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = new_m_i

        # Normalize and store output
        out = acc / l_i
        out_ptrs = out_ptr + batch_idx * stride_ob + head_idx * stride_oh + offs_d * stride_od
        tl.store(out_ptrs, out.to(q.dtype))


def paged_attention_decode_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: list[list[int]] | torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference implementation of paged attention decode for CPU fallback and parity checks.

    Args:
        q: (batch_size, num_heads, 1, head_dim) or (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: list of lists or 2D tensor of physical block IDs
        seq_lens: list or 1D tensor of active sequence lengths
        scale: softmax scaling factor (defaults to 1 / sqrt(head_dim))

    Returns:
        output: (batch_size, num_heads, 1, head_dim)
    """
    if q.dim() == 3:
        q = q.unsqueeze(2)  # (B, H, 1, D)

    batch_size, num_heads, _, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_heads // num_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    seq_lens_list = seq_lens.tolist() if isinstance(seq_lens, torch.Tensor) else list(seq_lens)
    block_tables_list = (
        block_tables.tolist() if isinstance(block_tables, torch.Tensor) else block_tables
    )

    outputs: list[torch.Tensor] = []

    for b in range(batch_size):
        cur_seq_len = seq_lens_list[b]
        cur_blocks = block_tables_list[b]
        q_b = q[b : b + 1]  # (1, num_heads, 1, head_dim)

        if cur_seq_len == 0 or not cur_blocks:
            outputs.append(torch.zeros_like(q_b))
            continue

        num_needed_blocks = (cur_seq_len + block_size - 1) // block_size
        active_blocks = cur_blocks[:num_needed_blocks]

        # Gather K and V for this sequence
        k_blocks = k_cache[active_blocks]
        v_blocks = v_cache[active_blocks]

        k_flat = k_blocks.view(-1, num_kv_heads, head_dim)[:cur_seq_len]
        v_flat = v_blocks.view(-1, num_kv_heads, head_dim)[:cur_seq_len]

        k_seq = k_flat.unsqueeze(0).transpose(1, 2)  # (1, num_kv_heads, cur_seq_len, D)
        v_seq = v_flat.unsqueeze(0).transpose(1, 2)

        # GQA expansion if needed
        if num_queries_per_kv > 1:
            k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=1)
            v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=1)

        scores = torch.matmul(q_b, k_seq.transpose(-2, -1)) * scale  # (1, H, 1, cur_seq_len)
        probs = F.softmax(scores, dim=-1)
        out_b = torch.matmul(probs, v_seq)  # (1, H, 1, D)
        outputs.append(out_b)

    return torch.cat(outputs, dim=0)


def paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: list[list[int]] | torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Dynamic dispatch decode attention: executes Triton GPU kernel or PyTorch reference.

    Args:
        q: (batch_size, num_heads, 1, head_dim) or (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: list of physical block IDs or 2D tensor
        seq_lens: list of context lengths or 1D tensor
        scale: softmax scaling factor

    Returns:
        output: (batch_size, num_heads, 1, head_dim)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    # GPU Triton path
    if HAS_TRITON and q.is_cuda and torch.cuda.is_available():
        q_3d = q.squeeze(2) if q.dim() == 4 else q

        batch_size, num_heads, head_dim = q_3d.shape
        block_size = k_cache.shape[1]

        if not isinstance(block_tables, torch.Tensor):
            max_blocks = max(len(bt) for bt in block_tables) if block_tables else 1
            padded_bt = [bt + [0] * (max_blocks - len(bt)) for bt in block_tables]
            bt_tensor = torch.tensor(padded_bt, dtype=torch.int32, device=q.device)
        else:
            bt_tensor = block_tables.to(dtype=torch.int32, device=q.device)

        if not isinstance(seq_lens, torch.Tensor):
            sl_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=q.device)
        else:
            sl_tensor = seq_lens.to(dtype=torch.int32, device=q.device)

        out = torch.empty_like(q_3d)
        grid = (batch_size, num_heads)

        _paged_attention_decode_kernel[grid](
            out,
            q_3d,
            k_cache,
            v_cache,
            bt_tensor,
            sl_tensor,
            scale,
            q_3d.stride(0),
            q_3d.stride(1),
            q_3d.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bt_tensor.stride(0),
            bt_tensor.stride(1),
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
        )
        return out.unsqueeze(2)

    # Fast PyTorch reference path
    return paged_attention_decode_ref(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
    )
