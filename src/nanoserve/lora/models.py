"""Multi-LoRA linear projection layer supporting heterogeneous batched adapter routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoraWeight:
    """Low-rank parameter matrices A and B for a specific linear layer."""

    lora_a: torch.Tensor  # shape: (r, in_features)
    lora_b: torch.Tensor  # shape: (out_features, r)
    scaling: float = 1.0


class LoraLinear(nn.Module):
    """Linear layer wrapping a frozen base projection with dynamic multi-adapter execution."""

    def __init__(self, base_linear: nn.Linear) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.adapters: dict[str, LoraWeight] = {}

    def add_adapter(self, name: str, weight: LoraWeight) -> None:
        """Register or update an adapter's low-rank weights."""
        self.adapters[name] = weight

    def remove_adapter(self, name: str) -> None:
        """Evict an adapter from this layer."""
        self.adapters.pop(name, None)

    def forward(
        self,
        x: torch.Tensor,
        adapter_names: list[str | None] | None = None,
    ) -> torch.Tensor:
        """Forward pass applying base projection and per-token LoRA deltas.

        Args:
            x: Input tensor of shape (batch_size, in_features) or (1, seq_len, in_features)
            adapter_names: Optional list of adapter names for each token/sequence in the batch

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Base model forward pass
        base_out = self.base_linear(x)

        if not self.adapters or adapter_names is None:
            return base_out

        # Flatten leading dimensions if 3D
        orig_shape = base_out.shape
        x_flat = x.view(-1, x.shape[-1])
        out_flat = base_out.view(-1, base_out.shape[-1]).clone()

        # If all tokens use the same adapter
        unique_adapters = set(adapter_names)
        if len(unique_adapters) == 1:
            name = next(iter(unique_adapters))
            if name is not None and name in self.adapters:
                w = self.adapters[name]
                delta = (x_flat @ w.lora_a.t()) @ w.lora_b.t() * w.scaling
                out_flat = out_flat + delta
            return out_flat.view(orig_shape)

        # Heterogeneous batch: apply adapter delta per token
        for idx, name in enumerate(adapter_names):
            if name is not None and name in self.adapters:
                w = self.adapters[name]
                token_x = x_flat[idx : idx + 1]
                delta = (token_x @ w.lora_a.t()) @ w.lora_b.t() * w.scaling
                out_flat[idx : idx + 1] = out_flat[idx : idx + 1] + delta

        return out_flat.view(orig_shape)
