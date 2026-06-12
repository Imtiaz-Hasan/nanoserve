"""Scheduler: iteration-level continuous batcher with preemption (swap and recompute)."""

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
        return (
            len(self.scheduled_seq_groups) == 0
            and not self.blocks_to_swap_in
            and not self.blocks_to_swap_out
        )


class Scheduler:
    """Continuous iteration-level batcher with preemption support.

    Maintains:
      - waiting: new requests needing initial prefill (or recomputed prefill)
      - running: active requests in decode phase
      - swapped: preempted requests swapped out to CPU memory
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
        self.num_preemptions: int = 0

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_swapped(self) -> int:
        return len(self._swapped)

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
        """Run one iteration-level scheduling step with preemption."""
        # 1. Clean up finished sequence groups
        still_running: list[SequenceGroup] = []
        for sg in self._running:
            if sg.is_finished:
                for seq in sg.sequences:
                    self.block_manager.free(seq)
            else:
                still_running.append(sg)
        self._running = still_running

        # 2. Check swapped queue: resume preempted requests (highest priority)
        if self._swapped and len(self._running) < self.config.max_num_seqs:
            sg = self._swapped[0]
            seq = sg.first_seq
            if self.block_manager.can_swap_in(seq):
                self._swapped.popleft()
                swap_in_map = self.block_manager.swap_in(seq)
                seq.status = SequenceStatus.RUNNING
                self._running.append(sg)
                return SchedulerOutputs(
                    scheduled_seq_groups=[],
                    is_prefill=False,
                    num_batched_tokens=0,
                    blocks_to_swap_in=list(swap_in_map.items()),
                )

        # 3. Admit new sequences from waiting queue for prefill
        if self._waiting and len(self._running) < self.config.max_num_seqs:
            scheduled_prefills: list[SequenceGroup] = []
            num_batched_tokens = 0

            while self._waiting and len(self._running) < self.config.max_num_seqs:
                sg = self._waiting[0]
                seq = sg.first_seq
                prompt_len = len(seq.all_token_ids)

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

        # 4. If running sequences exist, schedule batched decode (with preemption if OOM)
        if self._running:
            blocks_to_swap_out: list[tuple[int, int]] = []

            # Preemption loop: if any sequence cannot allocate next block, preempt victims (LIFO)
            i = 0
            while i < len(self._running):
                seq = self._running[i].first_seq
                if not self.block_manager.can_allocate(seq):
                    if len(self._running) <= 1:
                        break
                    victim_sg = self._running.pop()
                    victim_seq = victim_sg.first_seq
                    self.num_preemptions += 1

                    if self.config.preemption_mode == "swap" and self.block_manager.can_swap_out(
                        victim_seq
                    ):
                        swap_out_map = self.block_manager.swap_out(victim_seq)
                        victim_seq.status = SequenceStatus.SWAPPED
                        self._swapped.append(victim_sg)
                        blocks_to_swap_out.extend(list(swap_out_map.items()))
                    else:
                        self.block_manager.free(victim_seq)
                        victim_seq.status = SequenceStatus.WAITING
                        self._waiting.appendleft(victim_sg)

                    i = 0
                    continue
                i += 1

            scheduled_decodes: list[SequenceGroup] = []
            num_tokens = 0
            for sg in self._running:
                seq = sg.first_seq
                if self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    scheduled_decodes.append(sg)
                    num_tokens += 1
                    if len(scheduled_decodes) >= self.config.max_num_seqs:
                        break

            if scheduled_decodes or blocks_to_swap_out:
                return SchedulerOutputs(
                    scheduled_seq_groups=scheduled_decodes,
                    is_prefill=False,
                    num_batched_tokens=num_tokens,
                    blocks_to_swap_out=blocks_to_swap_out,
                )

        return SchedulerOutputs(
            scheduled_seq_groups=[],
            is_prefill=True,
            num_batched_tokens=0,
        )

    def has_unfinished_seqs(self) -> bool:
        """Whether there are any unfinished sequences in any queue."""
        return bool(self._waiting) or bool(self._running) or bool(self._swapped)
