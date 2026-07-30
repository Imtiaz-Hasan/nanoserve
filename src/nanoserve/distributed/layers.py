"""Megatron-LM style Tensor Parallel layers: ColumnParallel, RowParallel, and VocabParallel."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from nanoserve.distributed.communicator import CommunicatorProtocol


class ColumnParallelLinear(nn.Module):
    """Linear layer sharded along the output feature dimension (columns of W^T).

    Produces a slice of the output features without requiring communication.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        rank: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.rank = rank

        assert out_features % tp_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_size ({tp_size})"
        )
        self.local_out_features = out_features // tp_size

        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty(self.local_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without inter-worker communication."""
        return F.linear(x, self.weight, self.bias)

    @classmethod
    def from_full_linear(
        cls,
        full_linear: nn.Linear,
        rank: int,
        tp_size: int,
    ) -> ColumnParallelLinear:
        """Shard full nn.Linear weights across tensor parallel rank."""
        layer = cls(
            in_features=full_linear.in_features,
            out_features=full_linear.out_features,
            tp_size=tp_size,
            rank=rank,
            bias=full_linear.bias is not None,
        )
        chunk_size = full_linear.out_features // tp_size
        start = rank * chunk_size
        end = start + chunk_size

        with torch.no_grad():
            layer.weight.copy_(full_linear.weight[start:end])
            if full_linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(full_linear.bias[start:end])

        return layer


class RowParallelLinear(nn.Module):
    """Linear layer sharded along the input feature dimension (rows of W^T).

    Requires an all-reduce sum across TP workers to aggregate partial sums.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        rank: int = 0,
        communicator: CommunicatorProtocol | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.rank = rank
        self.communicator = communicator

        assert in_features % tp_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_size ({tp_size})"
        )
        self.local_in_features = in_features // tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all-reduce sum across ranks."""
        local_out = F.linear(x, self.weight)

        if self.communicator is not None and self.tp_size > 1:
            out = self.communicator.all_reduce(local_out)
        else:
            out = local_out

        if self.bias is not None:
            out = out + self.bias

        return out

    @classmethod
    def from_full_linear(
        cls,
        full_linear: nn.Linear,
        rank: int,
        tp_size: int,
        communicator: CommunicatorProtocol | None = None,
    ) -> RowParallelLinear:
        """Shard full nn.Linear weights across tensor parallel rank."""
        layer = cls(
            in_features=full_linear.in_features,
            out_features=full_linear.out_features,
            tp_size=tp_size,
            rank=rank,
            communicator=communicator,
            bias=full_linear.bias is not None,
        )
        chunk_size = full_linear.in_features // tp_size
        start = rank * chunk_size
        end = start + chunk_size

        with torch.no_grad():
            layer.weight.copy_(full_linear.weight[:, start:end])
            if full_linear.bias is not None and layer.bias is not None:
                # Bias is added once after all-reduce
                layer.bias.copy_(full_linear.bias)

        return layer


class VocabParallelEmbedding(nn.Module):
    """Embedding layer sharded along vocabulary size across TP ranks."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tp_size: int = 1,
        rank: int = 0,
        communicator: CommunicatorProtocol | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_size = tp_size
        self.rank = rank
        self.communicator = communicator

        self.vocab_start = (num_embeddings * rank) // tp_size
        self.vocab_end = (num_embeddings * (rank + 1)) // tp_size
        self.local_vocab_size = self.vocab_end - self.vocab_start

        self.weight = nn.Parameter(torch.empty(self.local_vocab_size, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lookup token embeddings and all-reduce across ranks."""
        # Create mask for tokens that belong to this rank
        valid_mask = (x >= self.vocab_start) & (x < self.vocab_end)
        local_indices = torch.clamp(x - self.vocab_start, 0, self.local_vocab_size - 1)

        local_embeds = F.embedding(local_indices, self.weight)

        # Zero out embeddings for tokens outside this rank's partition
        local_embeds = local_embeds * valid_mask.unsqueeze(-1).to(local_embeds.dtype)

        if self.communicator is not None and self.tp_size > 1:
            return self.communicator.all_reduce(local_embeds)

        return local_embeds

    @classmethod
    def from_full_embedding(
        cls,
        full_embedding: nn.Embedding,
        rank: int,
        tp_size: int,
        communicator: CommunicatorProtocol | None = None,
    ) -> VocabParallelEmbedding:
        """Shard full nn.Embedding weights across TP ranks."""
        layer = cls(
            num_embeddings=full_embedding.num_embeddings,
            embedding_dim=full_embedding.embedding_dim,
            tp_size=tp_size,
            rank=rank,
            communicator=communicator,
        )
        with torch.no_grad():
            layer.weight.copy_(full_embedding.weight[layer.vocab_start : layer.vocab_end])
        return layer
