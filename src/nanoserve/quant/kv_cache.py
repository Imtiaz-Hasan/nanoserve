"""Quantized KV Cache: INT8/FP8 symmetric block-level quantization and dequantization."""

from __future__ import annotations

import torch


def quantize_tensor_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor or per-channel INT8 quantization.

    Args:
        tensor: (..., head_dim) floating point tensor

    Returns:
        quantized: (..., head_dim) torch.int8 tensor
        scale: (...) float32 scaling factor
    """
    max_val = torch.max(torch.abs(tensor), dim=-1, keepdim=True).values
    scale = torch.clamp(max_val / 127.0, min=1e-8)
    quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def dequantize_tensor_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize INT8 tensor back to float32 using scale factors."""
    if scale.dim() < quantized.dim():
        scale = scale.unsqueeze(-1)
    return quantized.to(torch.float32) * scale


class QuantizedKVCache:
    """Physical Paged KV Cache storing 8-bit quantized tensors with block-level scale factors.

    Halves memory consumption from 2 bytes (FP16) or 4 bytes (FP32) to 1 byte per token value.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = torch.device(device)

        # 8-bit physical cache pools
        self.k_cache_quant = torch.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=torch.int8,
            device=self.device,
        )
        self.v_cache_quant = torch.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=torch.int8,
            device=self.device,
        )

        # Per-block, per-head scale factors
        self.k_scales = torch.ones(
            (num_blocks, block_size, num_kv_heads),
            dtype=torch.float32,
            device=self.device,
        )
        self.v_scales = torch.ones(
            (num_blocks, block_size, num_kv_heads),
            dtype=torch.float32,
            device=self.device,
        )

    def write_slots(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Quantize and write Key and Value tokens into physical cache slots.

        Args:
            key: (total_tokens, num_kv_heads, head_dim)
            value: (total_tokens, num_kv_heads, head_dim)
            slot_mapping: (total_tokens,) flat slot indices
        """
        k_quant, k_scale = quantize_tensor_int8(key)
        v_quant, v_scale = quantize_tensor_int8(value)

        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size

        self.k_cache_quant[block_indices, block_offsets] = k_quant
        self.v_cache_quant[block_indices, block_offsets] = v_quant
        self.k_scales[block_indices, block_offsets] = k_scale
        self.v_scales[block_indices, block_offsets] = v_scale

    def gather_and_dequantize(
        self,
        block_table: list[int],
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather physical blocks for a sequence and dequantize to float32 for attention.

        Args:
            block_table: List of physical block IDs
            num_tokens: Active total tokens

        Returns:
            gathered_k: (1, num_kv_heads, num_tokens, head_dim)
            gathered_v: (1, num_kv_heads, num_tokens, head_dim)
        """
        if not block_table or num_tokens <= 0:
            empty = torch.empty(
                (1, self.num_kv_heads, 0, self.head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            return empty, empty

        num_needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        active_blocks = block_table[:num_needed_blocks]

        # Gather quantized blocks
        k_blocks = self.k_cache_quant[active_blocks]  # (B_act, block_size, H, D)
        v_blocks = self.v_cache_quant[active_blocks]

        k_scales = self.k_scales[active_blocks]  # (B_act, block_size, H)
        v_scales = self.v_scales[active_blocks]

        # Flatten across block tokens
        k_flat = k_blocks.view(-1, self.num_kv_heads, self.head_dim)[:num_tokens]
        v_flat = v_blocks.view(-1, self.num_kv_heads, self.head_dim)[:num_tokens]

        k_scales_flat = k_scales.view(-1, self.num_kv_heads)[:num_tokens]
        v_scales_flat = v_scales.view(-1, self.num_kv_heads)[:num_tokens]

        # Dequantize
        k_dequant = dequantize_tensor_int8(k_flat, k_scales_flat)
        v_dequant = dequantize_tensor_int8(v_flat, v_scales_flat)

        # Reshape to (1, num_kv_heads, num_tokens, head_dim)
        gathered_k = k_dequant.unsqueeze(0).transpose(1, 2)
        gathered_v = v_dequant.unsqueeze(0).transpose(1, 2)

        return gathered_k, gathered_v
