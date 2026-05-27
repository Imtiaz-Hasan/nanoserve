"""Sequence and SequenceGroup: the units of work flowing through the engine."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoserve.sampling.params import SamplingParams


class SequenceStatus(enum.Enum):
    """Lifecycle states for a single sequence."""

    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH = "finished_length"
    FINISHED_ABORTED = "finished_aborted"

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class Sequence:
    """A single sequence being generated.

    Tracks prompt tokens, generated output tokens, and block-table metadata.
    """

    seq_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    logical_block_table: list[int] = field(default_factory=list)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int) -> None:
        """Append a generated token."""
        self.output_token_ids.append(token_id)

    def fork(self, new_seq_id: int) -> Sequence:
        """Create a copy of this sequence for beam search branching."""
        return Sequence(
            seq_id=new_seq_id,
            prompt_token_ids=list(self.prompt_token_ids),
            output_token_ids=list(self.output_token_ids),
            status=self.status,
            logical_block_table=list(self.logical_block_table),
        )


@dataclass
class SequenceGroup:
    """A group of sequences sharing a prompt, associated with one user request.

    In greedy / sampling mode this contains exactly one sequence.
    In beam search it would contain `best_of` sequences (future work).
    """

    request_id: str
    sequences: list[Sequence]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    prompt_token_ids: list[int] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        return all(seq.status.is_finished for seq in self.sequences)

    @property
    def first_seq(self) -> Sequence:
        return self.sequences[0]

    def get_unfinished_sequences(self) -> list[Sequence]:
        return [s for s in self.sequences if not s.status.is_finished]
