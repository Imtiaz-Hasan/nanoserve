"""Rotary Position Embeddings (RoPE) with optional scaling variants."""

from __future__ import annotations

import torch


class RotaryEmbedding:
    """Precomputed rotary embeddings applied to query and key tensors.

    Supports linear and NTK-aware RoPE scaling for extended context.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        device: torch.device | None = None,
        scaling_factor: float = 1.0,
    ) -> None:
        self.head_dim = head_dim
        self.max_position = max_position
        self.base = base
        self.scaling_factor = scaling_factor

        effective_base = base
        if scaling_factor != 1.0:
            # NTK-aware scaling: base^(dim/(dim-2)) → scale the base frequency
            effective_base = base * (scaling_factor ** (head_dim / (head_dim - 2)))

        inv_freq = 1.0 / (
            effective_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_position, dtype=torch.float32)
        if scaling_factor != 1.0:
            positions = positions / scaling_factor

        # compute outer product of positions and inv_freq
        freqs = torch.outer(positions, inv_freq)

        if device is not None:
            freqs = freqs.to(device)

        self._cos_cached = freqs.cos()  # (max_position, head_dim // 2)
        self._sin_cached = freqs.sin()  # (max_position, head_dim // 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to q and k.

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            positions: (batch, seq_len), (seq_len,), or (batch,) — position indices

        Returns:
            Rotated (q, k) with same shapes.
        """
        cos = self._cos_cached.to(q.device)
        sin = self._sin_cached.to(q.device)

        batch, _, seq_len, _ = q.shape

        if positions.dim() == 1:
            if positions.shape[0] == batch and seq_len == 1:
                # Batched decode: positions is (batch,)
                pos_2d = positions.unsqueeze(1)
            elif batch == 1:
                # Single sequence prefill: positions is (seq_len,)
                pos_2d = positions.unsqueeze(0)
            else:
                pos_2d = positions.view(batch, seq_len)
        else:
            pos_2d = positions

        cos_pos = cos[pos_2d]  # (batch, seq_len, head_dim // 2)
        sin_pos = sin[pos_2d]  # (batch, seq_len, head_dim // 2)

        # Insert head dimension for broadcasting: (batch, 1, seq_len, head_dim // 2)
        cos_pos = cos_pos.unsqueeze(1)
        sin_pos = sin_pos.unsqueeze(1)

        q_rot = _apply_rotary(q, cos_pos, sin_pos)
        k_rot = _apply_rotary(k, cos_pos, sin_pos)
        return q_rot, k_rot


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embedding to a single tensor.

    x: (..., head_dim) where head_dim is even.
    cos, sin: (..., head_dim // 2), broadcastable to x's shape.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
