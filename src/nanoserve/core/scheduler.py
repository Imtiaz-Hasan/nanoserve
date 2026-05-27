"""Scheduler: manages waiting, running, and swapped sequence queues.

Week 1: simple FCFS single-sequence scheduling.
Week 4 implements full continuous batching with budget enforcement.
"""

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
    num_prefill_groups: int
    num_decode_groups: int
    blocks_to_swap_in: list[tuple[int, int]] = field(default_factory=list)
    blocks_to_swap_out: list[tuple[int, int]] = field(default_factory=list)
    blocks_to_copy: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_seq_groups) == 0


class Scheduler:
    """Iteration-level scheduler managing sequence lifecycle.

    Week 1: processes one sequence at a time. No batching, no preemption.
    The interface is designed for Week 4's continuous batching upgrade.
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
        for queue in [self._running, list(self._waiting)]:
            for sg in queue:
                if sg.request_id == request_id:
                    for seq in sg.sequences:
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        self.block_manager.free(seq)
                    if sg in self._running:
                        self._running.remove(sg)
                    return

        # Also check waiting deque
        self._waiting = deque(sg for sg in self._waiting if sg.request_id != request_id)

    def schedule(self) -> SchedulerOutputs:
        """Run one scheduling iteration.

        Week 1: admits at most one waiting sequence if running is empty,
        or continues decoding the running sequence.
        """
        scheduled: list[SequenceGroup] = []
        num_prefill = 0
        num_decode = 0

        # Retire finished sequences
        still_running: list[SequenceGroup] = []
        for sg in self._running:
            if sg.is_finished:
                for seq in sg.sequences:
                    self.block_manager.free(seq)
            else:
                still_running.append(sg)
        self._running = still_running

        # Continue decoding running sequences
        for sg in self._running:
            # Extend blocks if needed for the next token
            for seq in sg.get_unfinished_sequences():
                if self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
            scheduled.append(sg)
            num_decode += 1

        # Admit new sequences from waiting (Week 1: one at a time)
        while self._waiting and len(self._running) < self.config.max_num_seqs:
            sg = self._waiting[0]
            seq = sg.first_seq

            if not self.block_manager.can_allocate(seq):
                break

            self._waiting.popleft()
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self._running.append(sg)
            scheduled.append(sg)
            num_prefill += 1

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled,
            num_prefill_groups=num_prefill,
            num_decode_groups=num_decode,
        )

    def has_unfinished_seqs(self) -> bool:
        """Whether there are any unfinished sequences in any queue."""
        return bool(self._waiting) or bool(self._running) or bool(self._swapped)
