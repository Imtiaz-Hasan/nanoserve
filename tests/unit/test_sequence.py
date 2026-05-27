"""Unit tests for Sequence and SequenceGroup lifecycle."""

from nanoserve.core.sequence import Sequence, SequenceGroup, SequenceStatus
from nanoserve.sampling.params import SamplingParams


def test_sequence_initial_state() -> None:
    """New sequences start in WAITING status with no output tokens."""
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3])
    assert seq.status == SequenceStatus.WAITING
    assert seq.num_prompt_tokens == 3
    assert seq.num_output_tokens == 0
    assert seq.num_total_tokens == 3
    assert seq.last_token_id == 3


def test_sequence_append_and_count() -> None:
    """Appending tokens updates counts and last_token_id."""
    seq = Sequence(seq_id=0, prompt_token_ids=[10, 20])
    seq.append_token(30)
    seq.append_token(40)

    assert seq.num_output_tokens == 2
    assert seq.num_total_tokens == 4
    assert seq.last_token_id == 40
    assert seq.all_token_ids == [10, 20, 30, 40]


def test_sequence_status_is_finished() -> None:
    """Finished statuses are correctly detected."""
    assert SequenceStatus.FINISHED_STOPPED.is_finished
    assert SequenceStatus.FINISHED_LENGTH.is_finished
    assert SequenceStatus.FINISHED_ABORTED.is_finished
    assert not SequenceStatus.WAITING.is_finished
    assert not SequenceStatus.RUNNING.is_finished
    assert not SequenceStatus.SWAPPED.is_finished


def test_sequence_fork() -> None:
    """Fork creates an independent copy."""
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3])
    seq.append_token(4)

    forked = seq.fork(new_seq_id=1)
    assert forked.seq_id == 1
    assert forked.all_token_ids == [1, 2, 3, 4]

    # Mutations are independent
    forked.append_token(5)
    assert seq.num_output_tokens == 1
    assert forked.num_output_tokens == 2


def test_sequence_group_lifecycle() -> None:
    """SequenceGroup tracks request identity and finish state."""
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2])
    params = SamplingParams(temperature=0.0, max_tokens=10)
    sg = SequenceGroup(
        request_id="req-001",
        sequences=[seq],
        sampling_params=params,
        prompt_token_ids=[1, 2],
    )

    assert not sg.is_finished
    assert sg.first_seq is seq
    assert len(sg.get_unfinished_sequences()) == 1

    seq.status = SequenceStatus.FINISHED_STOPPED
    assert sg.is_finished
    assert len(sg.get_unfinished_sequences()) == 0
