"""Scheduler: iteration-level continuous batcher with chunked prefill and mixed batches."""

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
    seq_chunk_lens: dict[int, int] = field(default_factory=dict)
    is_prefill: bool = False
    num_batched_tokens: int = 0
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
    """Continuous iteration-level batcher with chunked prefill & mixed batching (SARATHI-style).

    Maintains:
      - waiting: requests needing chunked prefill
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
        """Run one iteration-level scheduling step with chunked prefill and mixed batching."""
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

        scheduled_seq_groups: list[SequenceGroup] = []
        seq_chunk_lens: dict[int, int] = {}
        blocks_to_swap_out: list[tuple[int, int]] = []
        total_batched_tokens = 0
        remaining_budget = self.config.max_num_batched_tokens

        # 3. Schedule Running Decode Sequences (Priority 1 — No Starvation)
        if self._running:
            # Preemption loop: if any decode sequence cannot allocate next block, preempt victims
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
                        victim_seq.num_computed_tokens = 0
                        self._waiting.appendleft(victim_sg)

                    i = 0
                    continue
                i += 1

            for sg in self._running:
                seq = sg.first_seq
                if remaining_budget >= 1 and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    scheduled_seq_groups.append(sg)
                    seq_chunk_lens[seq.seq_id] = 1
                    total_batched_tokens += 1
                    remaining_budget -= 1
                    if len(scheduled_seq_groups) >= self.config.max_num_seqs:
                        break

        # 4. Schedule Prefill Chunks for Waiting Sequences (Priority 2 — In-Situ Chunking)
        while (
            remaining_budget > 0 and self._waiting and len(self._running) < self.config.max_num_seqs
        ):
            sg = self._waiting[0]
            seq = sg.first_seq

            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)

            uncomputed = len(seq.all_token_ids) - seq.num_computed_tokens
            if uncomputed <= 0:
                self._waiting.popleft()
                self._running.append(sg)
                seq.status = SequenceStatus.RUNNING
                continue

            chunk_len = min(remaining_budget, uncomputed)

            scheduled_seq_groups.append(sg)
            seq_chunk_lens[seq.seq_id] = chunk_len
            total_batched_tokens += chunk_len
            remaining_budget -= chunk_len

            if seq.num_computed_tokens + chunk_len >= len(seq.all_token_ids):
                # Prefill complete: transition sequence to running
                self._waiting.popleft()
                self._running.append(sg)
                seq.status = SequenceStatus.RUNNING
            else:
                # Intermediate chunk: remains in waiting queue for next iteration
                break

        is_prefill = any(
            seq_chunk_lens.get(sg.first_seq.seq_id, 0) > 1 for sg in scheduled_seq_groups
        )

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            seq_chunk_lens=seq_chunk_lens,
            is_prefill=is_prefill,
            num_batched_tokens=total_batched_tokens,
            blocks_to_swap_out=blocks_to_swap_out,
        )

    def has_unfinished_seqs(self) -> bool:
        """Whether there are any unfinished sequences in any queue."""
        return bool(self._waiting) or bool(self._running) or bool(self._swapped)
