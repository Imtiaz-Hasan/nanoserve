"""Unit tests for Megatron-LM style Tensor Parallel layers and collective communicators."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from nanoserve.distributed.communicator import MockDistributedCommunicator
from nanoserve.distributed.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)


def test_column_parallel_linear_parity() -> None:
    """Verify ColumnParallelLinear computes exact output slices with 0 communication."""
    torch.manual_seed(42)
    in_features = 32
    out_features = 64
    tp_size = 2

    full = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(4, in_features)

    expected_full = full(x)

    # Rank 0 (first 32 output features)
    col_0 = ColumnParallelLinear.from_full_linear(full, rank=0, tp_size=tp_size)
    out_0 = col_0(x)
    assert torch.allclose(out_0, expected_full[:, :32], atol=1e-6)

    # Rank 1 (second 32 output features)
    col_1 = ColumnParallelLinear.from_full_linear(full, rank=1, tp_size=tp_size)
    out_1 = col_1(x)
    assert torch.allclose(out_1, expected_full[:, 32:], atol=1e-6)


def test_row_parallel_linear_all_reduce_parity() -> None:
    """Verify RowParallelLinear with All-Reduce sum matches full monolithic linear projection."""
    torch.manual_seed(42)
    in_features = 64
    out_features = 32
    tp_size = 2

    full = nn.Linear(in_features, out_features, bias=True)
    x = torch.randn(4, in_features)
    expected = full(x)

    shared_state: dict[str, list[torch.Tensor]] = {}
    comm_0 = MockDistributedCommunicator(rank=0, tp_size=tp_size, shared_state=shared_state)
    comm_1 = MockDistributedCommunicator(rank=1, tp_size=tp_size, shared_state=shared_state)

    row_0 = RowParallelLinear.from_full_linear(full, rank=0, tp_size=tp_size, communicator=comm_0)
    row_1 = RowParallelLinear.from_full_linear(full, rank=1, tp_size=tp_size, communicator=comm_1)

    # Input slices
    x_0 = x[:, :32]
    x_1 = x[:, 32:]

    # Phase 1: Publish partial sums to mock communicator
    _ = row_0(x_0)
    _ = row_1(x_1)

    # Phase 2: Read synchronized all-reduced result
    out_0 = row_0(x_0)
    out_1 = row_1(x_1)

    assert torch.allclose(out_0, expected, atol=1e-6)
    assert torch.allclose(out_1, expected, atol=1e-6)


def test_vocab_parallel_embedding_parity() -> None:
    """Verify VocabParallelEmbedding partitions vocabulary rows and reconstructs full embeddings."""
    torch.manual_seed(42)
    num_embeddings = 100
    embedding_dim = 16
    tp_size = 2

    full_embed = nn.Embedding(num_embeddings, embedding_dim)
    input_ids = torch.tensor([[5, 75, 12, 99], [0, 50, 49, 88]], dtype=torch.long)

    expected = full_embed(input_ids)

    shared_state: dict[str, list[torch.Tensor]] = {}
    comm_0 = MockDistributedCommunicator(rank=0, tp_size=tp_size, shared_state=shared_state)
    comm_1 = MockDistributedCommunicator(rank=1, tp_size=tp_size, shared_state=shared_state)

    vocab_0 = VocabParallelEmbedding.from_full_embedding(full_embed, rank=0, tp_size=tp_size, communicator=comm_0)
    vocab_1 = VocabParallelEmbedding.from_full_embedding(full_embed, rank=1, tp_size=tp_size, communicator=comm_1)

    # Phase 1: publish shard embeddings
    _ = vocab_0(input_ids)
    _ = vocab_1(input_ids)

    # Phase 2: collect synchronized embeddings
    out_0 = vocab_0(input_ids)
    out_1 = vocab_1(input_ids)

    assert torch.allclose(out_0, expected, atol=1e-6)
    assert torch.allclose(out_1, expected, atol=1e-6)


def test_end_to_end_parallel_mlp_parity() -> None:
    """Verify 2-layer sharded MLP block (ColumnParallel gate/up -> RowParallel down) matches baseline."""
    torch.manual_seed(42)
    hidden_size = 32
    intermediate_size = 64
    tp_size = 2

    # Baseline Monolithic MLP: SwiGLU = (x @ W_gate * silu(x @ W_up)) @ W_down
    w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
    w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
    w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    x = torch.randn(2, hidden_size)

    # Monolithic forward
    expected_mlp = w_down(F.silu(w_gate(x)) * w_up(x))

    # Tensor Parallel Sharding (TP=2)
    shared_state: dict[str, list[torch.Tensor]] = {}
    comm_0 = MockDistributedCommunicator(rank=0, tp_size=tp_size, shared_state=shared_state)
    comm_1 = MockDistributedCommunicator(rank=1, tp_size=tp_size, shared_state=shared_state)

    # Worker 0 layers
    gate_0 = ColumnParallelLinear.from_full_linear(w_gate, rank=0, tp_size=tp_size)
    up_0 = ColumnParallelLinear.from_full_linear(w_up, rank=0, tp_size=tp_size)
    down_0 = RowParallelLinear.from_full_linear(w_down, rank=0, tp_size=tp_size, communicator=comm_0)

    # Worker 1 layers
    gate_1 = ColumnParallelLinear.from_full_linear(w_gate, rank=1, tp_size=tp_size)
    up_1 = ColumnParallelLinear.from_full_linear(w_up, rank=1, tp_size=tp_size)
    down_1 = RowParallelLinear.from_full_linear(w_down, rank=1, tp_size=tp_size, communicator=comm_1)

    act_0 = F.silu(gate_0(x)) * up_0(x)  # (2, 32)
    act_1 = F.silu(gate_1(x)) * up_1(x)  # (2, 32)

    # Phase 1: publish partial down-projections
    _ = down_0(act_0)
    _ = down_1(act_1)

    # Phase 2: collect synchronized all-reduced outputs
    out_0 = down_0(act_0)
    out_1 = down_1(act_1)

    assert torch.allclose(out_0, expected_mlp, atol=1e-6)
    assert torch.allclose(out_1, expected_mlp, atol=1e-6)
