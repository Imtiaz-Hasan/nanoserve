"""Unit tests for LoraLinear (base passthrough, single-adapter, and batched routing)."""

import torch
from torch import nn

from nanoserve.lora.models import LoraLinear, LoraWeight


def test_lora_linear_base_passthrough() -> None:
    """Verify LoraLinear matches standard nn.Linear when no adapters are active."""
    torch.manual_seed(42)
    in_features = 32
    out_features = 64
    batch_size = 4

    base = nn.Linear(in_features, out_features, bias=False)
    lora_layer = LoraLinear(base)

    x = torch.randn(batch_size, in_features)

    base_out = base(x)
    lora_out_none = lora_layer(x)
    lora_out_empty = lora_layer(x, adapter_names=[None, None, None, None])

    assert torch.allclose(base_out, lora_out_none, atol=1e-6)
    assert torch.allclose(base_out, lora_out_empty, atol=1e-6)


def test_lora_linear_single_adapter_parity() -> None:
    """Verify single-adapter forward pass mathematically matches manual Low-Rank decomposition."""
    torch.manual_seed(42)
    in_features = 16
    out_features = 32
    rank = 4
    scaling = 2.0

    base = nn.Linear(in_features, out_features, bias=False)
    lora_layer = LoraLinear(base)

    a = torch.randn(rank, in_features)
    b = torch.randn(out_features, rank)

    weight = LoraWeight(lora_a=a, lora_b=b, scaling=scaling)
    lora_layer.add_adapter("sql_adapter", weight)

    x = torch.randn(2, in_features)

    # Manual LoRA: Y = X W^T + s * (X A^T) B^T
    expected = base(x) + scaling * ((x @ a.t()) @ b.t())
    actual = lora_layer(x, adapter_names=["sql_adapter", "sql_adapter"])

    assert torch.allclose(expected, actual, atol=1e-6)


def test_lora_linear_heterogeneous_batch() -> None:
    """Verify heterogeneous batch routes sequences to different adapters in 1 forward pass."""
    torch.manual_seed(42)
    in_features = 16
    out_features = 32
    rank = 4

    base = nn.Linear(in_features, out_features, bias=False)
    lora_layer = LoraLinear(base)

    # Adapter A
    a1 = torch.randn(rank, in_features)
    b1 = torch.randn(out_features, rank)
    lora_layer.add_adapter("code_lora", LoraWeight(lora_a=a1, lora_b=b1, scaling=1.5))

    # Adapter B
    a2 = torch.randn(rank, in_features)
    b2 = torch.randn(out_features, rank)
    lora_layer.add_adapter("chat_lora", LoraWeight(lora_a=a2, lora_b=b2, scaling=0.5))

    # Batch of 3 tokens: [token0 -> code_lora, token1 -> chat_lora, token2 -> base]
    x = torch.randn(3, in_features)
    adapter_names = ["code_lora", "chat_lora", None]

    out = lora_layer(x, adapter_names=adapter_names)

    # Verify token 0 has code_lora delta
    exp_0 = base(x[0:1]) + 1.5 * ((x[0:1] @ a1.t()) @ b1.t())
    assert torch.allclose(out[0:1], exp_0, atol=1e-6)

    # Verify token 1 has chat_lora delta
    exp_1 = base(x[1:2]) + 0.5 * ((x[1:2] @ a2.t()) @ b2.t())
    assert torch.allclose(out[1:2], exp_1, atol=1e-6)

    # Verify token 2 has pure base output (no delta)
    exp_2 = base(x[2:3])
    assert torch.allclose(out[2:3], exp_2, atol=1e-6)
