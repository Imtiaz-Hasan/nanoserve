"""Scheduler: iteration-level continuous batcher managing waiting, running, and swapped queues."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from nanoserve.config import SchedulerConfig
from nanoserve.core.block_manager import BlockManager
from nanoserve.core.sequence import SequenceGroup, SequenceStatus


@dataclass
class SchedulerOutputs:
    """What the scheduler tells the model runner to execute this step."""

    scheduled_seq_groups: list[SequenceGroup]
    is_prefill: bool
    num_batched_tokens: int
    blocks_to_swap_in: list[tuple[int, int]] = field(default_factory=list)
    blocks_to_swap_out: list[tuple[int, int]] = field(default_factory=list)
    blocks_to_copy: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_seq_groups) == 0


class Scheduler:
    """Continuous iteration-level batcher (Orca style, Yu et al., OSDI 2022).

    Maintains:
      - waiting: new requests needing prefill
      - running: active requests in decode phase
      - swapped: preempted requests (swapped to CPU memory)
    """

    def __init__(
        self,
        config: SchedulerConfig,
        block_manager: BlockManager,
    ) -> None:
        self.config = config
        self.block_manager = block_manager
        self._waiting: deque[SequenceGroup] = deque()
        self._running: list[SequenceGroup] = []
        self._swapped: deque[SequenceGroup] = deque()

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        """Add a new request to the waiting queue."""
        self._waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        """Abort and free a request by ID."""
        for queue in [self._running, list(self._waiting), list(self._swapped)]:
            for sg in queue:
                if sg.request_id == request_id:
                    for seq in sg.sequences:
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.block_manager.free(seq)
                    if sg in self._running:
                        self._running.remove(sg)
                    return

        self._waiting = deque(sg for sg in self._waiting if sg.request_id != request_id)
        self._swapped = deque(sg for sg in self._swapped if sg.request_id != request_id)

    def schedule(self) -> SchedulerOutputs:
        """Run one iteration-level scheduling step."""
        # 1. Clean up finished sequence groups
        still_running: list[SequenceGroup] = []
        for sg in self._running:
            if sg.is_finished:
                for seq in sg.sequences:
                    self.block_manager.free(seq)
            else:
                still_running.append(sg)
        self._running = still_running

        # 2. If waiting requests exist and capacity allows, schedule prefill first
        if self._waiting and len(self._running) < self.config.max_num_seqs:
            scheduled_prefills: list[SequenceGroup] = []
            num_batched_tokens = 0

            while self._waiting and len(self._running) < self.config.max_num_seqs:
                sg = self._waiting[0]
                seq = sg.first_seq
                prompt_len = len(seq.prompt_token_ids)

                if num_batched_tokens + prompt_len > self.config.max_num_batched_tokens:
                    break

                if not self.block_manager.can_allocate(seq):
                    break

                self._waiting.popleft()
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self._running.append(sg)
                scheduled_prefills.append(sg)
                num_batched_tokens += prompt_len

            if scheduled_prefills:
                return SchedulerOutputs(
                    scheduled_seq_groups=scheduled_prefills,
                    is_prefill=True,
                    num_batched_tokens=num_batched_tokens,
                )

        # 3. Schedule all running sequences for batched decode
        if self._running:
            scheduled_decodes: list[SequenceGroup] = []
            num_tokens = 0
            for sg in self._running:
                seq = sg.first_seq
                # Ensure sequence has block capacity for next decode token
                if self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    scheduled_decodes.append(sg)
                    num_tokens += 1
                    if len(scheduled_decodes) >= self.config.max_num_seqs:
                        break
                else:
                    break

            if scheduled_decodes:
                return SchedulerOutputs(
                    scheduled_seq_groups=scheduled_decodes,
                    is_prefill=False,
                    num_batched_tokens=num_tokens,
                )

        return SchedulerOutputs(
            scheduled_seq_groups=[],
            is_prefill=True,
            num_batched_tokens=0,
        )

    def has_unfinished_seqs(self) -> bool:
        """Whether there are any unfinished sequences in any queue."""
        return bool(self._waiting) or bool(self._running) or bool(self._swapped)
