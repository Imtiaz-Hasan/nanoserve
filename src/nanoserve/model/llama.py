"""Llama-family decoder-only transformer: RMSNorm, SwiGLU, GQA, full forward pass."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from nanoserve.config import ModelConfig
from nanoserve.kernels.reshape_cache import gather_paged_kv, reshape_and_cache
from nanoserve.model.attention import reference_attention
from nanoserve.model.rope import RotaryEmbedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class LlamaAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) and RoPE."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_tables: list[list[int]] | None = None,
        seq_lens: list[int] | None = None,
        seq_token_ranges: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Self-attention forward pass with mixed prefill/decode and paged KV cache.

        Args:
            hidden_states: (batch, seq_len, hidden_size) or (1, total_tokens, hidden_size)
            positions: (batch, seq_len) or (total_tokens,)
            rope: RotaryEmbedding instance
            kv_cache: (k_cache, v_cache) physical paged cache tensors
            slot_mapping: (total_tokens,) slot mapping into physical cache
            block_tables: per-sequence physical block tables
            seq_lens: per-sequence active context lengths
            seq_token_ranges: list of (start_idx, end_idx) per sequence in the batch

        Returns:
            output: (batch, seq_len, hidden_size)
            updated_kv_cache: (k_cache, v_cache)
        """
        batch, seq_len, _ = hidden_states.shape
        total_tokens = batch * seq_len

        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Transpose to (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = rope.forward(q, k, positions)

        # Update and gather KV cache
        if (
            kv_cache is not None
            and slot_mapping is not None
            and block_tables is not None
            and seq_lens is not None
        ):
            k_cache, v_cache = kv_cache
            # 1. Scatter all new tokens into physical paged cache
            reshape_and_cache(k, v, k_cache, v_cache, slot_mapping)

            if seq_token_ranges is None:
                if batch == 1:
                    seq_token_ranges = [(0, seq_len)]
                else:
                    seq_token_ranges = [(i, i + 1) for i in range(batch)]

            # Flatten q across batch if needed: (1, num_heads, total_tokens, head_dim)
            if q.dim() == 4 and q.shape[0] > 1:
                q_flat = q.transpose(1, 2).reshape(1, total_tokens, self.num_heads, self.head_dim)
                q_flat = q_flat.transpose(1, 2)
            else:
                q_flat = q

            attn_outputs: list[torch.Tensor] = []
            for b, (start_idx, end_idx) in enumerate(seq_token_ranges):
                chunk_len = end_idx - start_idx
                q_b = q_flat[:, :, start_idx:end_idx, :]  # (1, num_heads, chunk_len, head_dim)
                gathered_k, gathered_v = gather_paged_kv(
                    k_cache, v_cache, block_tables[b], seq_lens[b]
                )

                if chunk_len == 1:
                    attn_b = reference_attention(q_b, gathered_k, gathered_v, scale=self.scale)
                else:
                    # Chunked prefill: causal masking within chunk + past attention
                    past_tokens = seq_lens[b] - chunk_len
                    total_kv = seq_lens[b]
                    q_indices = torch.arange(chunk_len, device=q.device).unsqueeze(1) + past_tokens
                    k_indices = torch.arange(total_kv, device=q.device).unsqueeze(0)
                    causal_mask = k_indices <= q_indices  # (chunk_len, total_kv)

                    scores = torch.matmul(q_b, gathered_k.transpose(-2, -1)) * self.scale
                    scores = scores.masked_fill(
                        ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )
                    probs = torch.softmax(scores, dim=-1)
                    attn_b = torch.matmul(probs, gathered_v)

                attn_outputs.append(attn_b)

            attn_output = torch.cat(attn_outputs, dim=2)

        elif kv_cache is not None:
            k_prev, v_prev = kv_cache
            k_cache = torch.cat([k_prev, k], dim=2)
            v_cache = torch.cat([v_prev, v], dim=2)
            attn_output = reference_attention(q, k_cache, v_cache, scale=self.scale)
            kv_cache = (k_cache, v_cache)
        else:
            attn_output = reference_attention(q, k, v, scale=self.scale)
            kv_cache = (k, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output: torch.Tensor = self.o_proj(attn_output)

        return output, kv_cache


class LlamaMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer: attention + MLP with residual connections."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_tables: list[list[int]] | None = None,
        seq_lens: list[int] | None = None,
        seq_token_ranges: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            rope=rope,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            seq_lens=seq_lens,
            seq_token_ranges=seq_token_ranges,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv


class LlamaForCausalLM(nn.Module):
    """Llama-family decoder-only causal language model.

    Embedding → N decoder layers → RMSNorm → LM head.
    Supports tied word embeddings (lm_head shares embedding weights).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, i) for i in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size)

        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_position=config.max_model_len,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        slot_mapping: torch.Tensor | None = None,
        block_tables: list[list[int]] | None = None,
        seq_lens: list[int] | None = None,
        seq_token_ranges: list[tuple[int, int]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with support for paged KV caches and mixed batches.

        Args:
            input_ids: (batch, seq_len) or (1, total_tokens) token ids
            positions: (batch, seq_len) or (total_tokens,) position indices
            kv_caches: list of per-layer (k_cache, v_cache) physical tensors
            slot_mapping: (total_tokens,) flat slot mappings for physical scatter
            block_tables: per-sequence physical block table IDs
            seq_lens: per-sequence active total lengths
            seq_token_ranges: per-sequence (start_idx, end_idx) in flattened batch

        Returns:
            logits: (batch, seq_len, vocab_size) or (1, total_tokens, vocab_size)
            new_kv_caches: updated per-layer KV caches
        """
        hidden_states = self.embed_tokens(input_ids)
        new_kv_caches: list[tuple[Any, Any]] = []

        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(
                hidden_states=hidden_states,
                positions=positions,
                rope=self.rope,
                kv_cache=layer_kv,
                slot_mapping=slot_mapping,
                block_tables=block_tables,
                seq_lens=seq_lens,
                seq_token_ranges=seq_token_ranges,
            )
            if new_kv is not None:
                new_kv_caches.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits: torch.Tensor = self.lm_head(hidden_states)

        return logits, new_kv_caches
