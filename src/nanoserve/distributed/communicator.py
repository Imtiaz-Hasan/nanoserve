"""Distributed Communicator: collective communication wrapper for PyTorch and mock."""

from __future__ import annotations

from typing import Protocol

import torch
import torch.distributed as dist


class CommunicatorProtocol(Protocol):
    """Protocol for distributed collective communication."""

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform an all-reduce (sum) across all ranks in the group."""
        ...

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors from all ranks into a list."""
        ...

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src rank to all ranks."""
        ...


class MockDistributedCommunicator:
    """In-process mock communicator for testing multi-rank tensor parallel layers."""

    def __init__(
        self, rank: int, tp_size: int, shared_state: dict[str, list[torch.Tensor]]
    ) -> None:
        self.rank = rank
        self.tp_size = tp_size
        self._shared_state = shared_state

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum tensor across all simulated ranks."""
        key = f"all_reduce_{tensor.shape}_{id(self._shared_state)}"
        if key not in self._shared_state:
            self._shared_state[key] = [torch.zeros_like(tensor) for _ in range(self.tp_size)]

        self._shared_state[key][self.rank] = tensor.clone()

        total = torch.zeros_like(tensor)
        for t in self._shared_state[key]:
            total = total + t
        return total

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Gather tensors across all ranks."""
        key = f"all_gather_{tensor.shape}_{id(self._shared_state)}"
        if key not in self._shared_state:
            self._shared_state[key] = [torch.zeros_like(tensor) for _ in range(self.tp_size)]

        self._shared_state[key][self.rank] = tensor.clone()
        return [t.clone() for t in self._shared_state[key]]

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from src rank."""
        key = f"broadcast_{tensor.shape}_{src}_{id(self._shared_state)}"
        if key not in self._shared_state:
            self._shared_state[key] = [torch.zeros_like(tensor) for _ in range(self.tp_size)]

        if self.rank == src:
            for r in range(self.tp_size):
                self._shared_state[key][r] = tensor.clone()

        return self._shared_state[key][self.rank].clone()


class PyTorchDistributedCommunicator:
    """Real PyTorch distributed communicator wrapping NCCL or Gloo backend."""

    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        self.group = group

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform real in-place all-reduce sum."""
        out = tensor.clone()
        if dist.is_initialized():
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.group)
        return out

    def all_gather(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Perform real all-gather."""
        if not dist.is_initialized():
            return [tensor.clone()]
        world_size = dist.get_world_size(self.group)
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group=self.group)
        return tensor_list

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Perform real broadcast from src."""
        out = tensor.clone()
        if dist.is_initialized():
            dist.broadcast(out, src=src, group=self.group)
        return out
