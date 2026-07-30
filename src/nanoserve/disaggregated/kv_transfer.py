"""KV Transfer Protocol: serialization and network transport for remote KV cache handoff."""

from __future__ import annotations

import io
from dataclasses import dataclass

import torch


@dataclass
class KVTransferPayload:
    """Container for serialized KV cache state transferred from Prefill to Decode worker."""

    request_id: str
    prompt_token_ids: list[int]
    first_token_id: int
    k_blocks: torch.Tensor  # (num_blocks, block_size, num_kv_heads, head_dim)
    v_blocks: torch.Tensor  # (num_blocks, block_size, num_kv_heads, head_dim)
    num_tokens: int

    def to_bytes(self) -> bytes:
        """Serialize payload into portable binary format."""
        buf = io.BytesIO()
        data = {
            "request_id": self.request_id,
            "prompt_token_ids": self.prompt_token_ids,
            "first_token_id": self.first_token_id,
            "k_blocks": self.k_blocks.cpu(),
            "v_blocks": self.v_blocks.cpu(),
            "num_tokens": self.num_tokens,
        }
        torch.save(data, buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data_bytes: bytes, device: str = "cpu") -> KVTransferPayload:
        """Deserialize binary buffer back into KVTransferPayload."""
        buf = io.BytesIO(data_bytes)
        data = torch.load(buf, weights_only=False)
        return cls(
            request_id=data["request_id"],
            prompt_token_ids=data["prompt_token_ids"],
            first_token_id=data["first_token_id"],
            k_blocks=data["k_blocks"].to(device),
            v_blocks=data["v_blocks"].to(device),
            num_tokens=data["num_tokens"],
        )
