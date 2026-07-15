"""Unit tests for INT8/FP8 Quantized KV Cache (precision, scatter/gather, and attention parity)."""

import torch
import torch.nn.functional as F  # noqa: N812

from nanoserve.model.attention import reference_attention
from nanoserve.quant.kv_cache import (
    QuantizedKVCache,
    dequantize_tensor_int8,
    quantize_tensor_int8,
)


def test_tensor_int8_quantization_dequantization_precision() -> None:
    """Verify INT8 symmetric quantization preserves high cosine similarity (>0.99)."""
    torch.manual_seed(42)
    original = torch.randn(16, 4, 64, dtype=torch.float32)

    quantized, scale = quantize_tensor_int8(original)
    assert quantized.dtype == torch.int8
    assert quantized.shape == original.shape

    dequantized = dequantize_tensor_int8(quantized, scale)
    cos_sim = F.cosine_similarity(original.view(-1, 64), dequantized.view(-1, 64), dim=-1)

    assert cos_sim.mean().item() > 0.995


def test_quantized_kv_cache_scatter_gather_roundtrip() -> None:
    """Verify QuantizedKVCache writes slots and gathers dequantized blocks accurately."""
    torch.manual_seed(42)
    num_blocks = 8
    block_size = 4
    num_kv_heads = 2
    head_dim = 32

    cache = QuantizedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
    )

    # 6 tokens distributed across block 3 (slots 12..15) and block 1 (slots 4..5)
    slot_mapping = torch.tensor([12, 13, 14, 15, 4, 5], dtype=torch.long)
    key = torch.randn(6, num_kv_heads, head_dim)
    value = torch.randn(6, num_kv_heads, head_dim)

    cache.write_slots(key, value, slot_mapping)

    block_table = [3, 1]
    gathered_k, gathered_v = cache.gather_and_dequantize(block_table, num_tokens=6)

    assert gathered_k.shape == (1, num_kv_heads, 6, head_dim)
    assert gathered_v.shape == (1, num_kv_heads, 6, head_dim)

    # Cosine similarity between original and gathered
    k_orig_t = key.unsqueeze(0).transpose(1, 2)
    v_orig_t = value.unsqueeze(0).transpose(1, 2)

    k_sim = F.cosine_similarity(k_orig_t.reshape(-1, head_dim), gathered_k.reshape(-1, head_dim))
    v_sim = F.cosine_similarity(v_orig_t.reshape(-1, head_dim), gathered_v.reshape(-1, head_dim))

    assert k_sim.mean().item() > 0.99
    assert v_sim.mean().item() > 0.99


def test_quantized_kv_cache_attention_parity() -> None:
    """Verify attention scores computed from quantized KV cache match reference SDPA (<0.05 MAE)."""
    torch.manual_seed(42)
    num_blocks = 4
    block_size = 4
    num_heads = 4
    num_kv_heads = 4
    head_dim = 32
    seq_len = 8

    cache = QuantizedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
    )

    key = torch.randn(seq_len, num_kv_heads, head_dim)
    value = torch.randn(seq_len, num_kv_heads, head_dim)
    slot_mapping = torch.arange(seq_len, dtype=torch.long)

    cache.write_slots(key, value, slot_mapping)

    q = torch.randn(1, num_heads, 1, head_dim)

    # 1. Attention with dequantized KV
    gathered_k, gathered_v = cache.gather_and_dequantize(block_table=[0, 1], num_tokens=seq_len)
    actual_out = reference_attention(q, gathered_k, gathered_v)

    # 2. Reference attention with unquantized float32 KV
    k_ref = key.unsqueeze(0).transpose(1, 2)
    v_ref = value.unsqueeze(0).transpose(1, 2)
    expected_out = reference_attention(q, k_ref, v_ref)

    mae = (actual_out - expected_out).abs().mean().item()
    assert mae < 0.05
