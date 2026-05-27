"""Unit tests for sampling: greedy, parameter validation, stop string detection."""

import pytest
import torch

from nanoserve.sampling.params import SamplingParams
from nanoserve.sampling.sampler import Sampler
from nanoserve.sampling.stop import StopChecker

# --- SamplingParams validation ---


def test_sampling_params_defaults() -> None:
    """Default params are valid."""
    params = SamplingParams()
    assert params.temperature == 1.0
    assert params.max_tokens == 256
    assert not params.is_greedy


def test_sampling_params_greedy() -> None:
    """temperature=0 is detected as greedy."""
    params = SamplingParams(temperature=0.0)
    assert params.is_greedy


def test_sampling_params_invalid_temperature() -> None:
    """Negative temperature is rejected."""
    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(temperature=-1.0)


def test_sampling_params_invalid_top_p() -> None:
    """top_p outside [0, 1] is rejected."""
    with pytest.raises(ValueError, match="top_p"):
        SamplingParams(top_p=1.5)


def test_sampling_params_invalid_max_tokens() -> None:
    """max_tokens < 1 is rejected."""
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingParams(max_tokens=0)


# --- Sampler ---


def test_greedy_sampling() -> None:
    """Greedy sampling returns argmax."""
    sampler = Sampler()
    logits = torch.tensor([[0.1, 0.5, 0.9, 0.2]])  # argmax = index 2
    params = SamplingParams(temperature=0.0)

    result = sampler.sample(logits, [params])
    assert result == [2]


def test_greedy_sampling_batch() -> None:
    """Batched greedy sampling returns per-row argmax."""
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.3],
            [0.2, 0.3, 0.7],
        ]
    )
    params = [SamplingParams(temperature=0.0)] * 3

    result = sampler.sample(logits, params)
    assert result == [1, 0, 2]


def test_sampler_batch_mismatch() -> None:
    """Mismatched batch size raises ValueError."""
    sampler = Sampler()
    logits = torch.tensor([[0.1, 0.5]])
    params = [SamplingParams(), SamplingParams()]

    with pytest.raises(ValueError, match="Batch size mismatch"):
        sampler.sample(logits, params)


def test_logit_bias() -> None:
    """Logit bias shifts the argmax."""
    sampler = Sampler()
    # Without bias: argmax = 1
    logits = torch.tensor([[0.1, 0.9, 0.2]])
    params = SamplingParams(temperature=0.0, logit_bias={2: 10.0})

    result = sampler.sample(logits.clone(), [params])
    assert result == [2]  # bias makes token 2 win


# --- StopChecker ---


def test_stop_token_detection() -> None:
    """Stop token IDs are detected."""
    checker = StopChecker(stop_strings=[], stop_token_ids=[0, 50])
    assert checker.should_stop_token(0)
    assert checker.should_stop_token(50)
    assert not checker.should_stop_token(1)


def test_stop_string_detection() -> None:
    """Stop strings at the end of text are detected."""
    checker = StopChecker(stop_strings=["</s>", "\n\n"], stop_token_ids=[])
    assert checker.should_stop_string("Hello world</s>") == "</s>"
    assert checker.should_stop_string("Paragraph one\n\n") == "\n\n"
    assert checker.should_stop_string("Hello world") is None


def test_stop_string_boundary_spanning() -> None:
    """Stop strings that accumulate across token boundaries are caught."""
    checker = StopChecker(stop_strings=["world"], stop_token_ids=[])

    # Simulate incremental text: "wor" + "ld"
    text = "Hello wor"
    assert checker.should_stop_string(text) is None

    text = "Hello world"
    assert checker.should_stop_string(text) == "world"


def test_stop_partial_match() -> None:
    """Partial match detection for output buffering."""
    checker = StopChecker(stop_strings=["STOP"], stop_token_ids=[])

    assert checker.check_partial_match("textST")  # "ST" is prefix of "STOP"
    assert checker.check_partial_match("textSTO")  # "STO" is prefix of "STOP"
    assert not checker.check_partial_match("textXYZ")


def test_empty_stop_string_ignored() -> None:
    """Empty stop strings don't cause false positives."""
    checker = StopChecker(stop_strings=["", "end"], stop_token_ids=[])
    assert checker.should_stop_string("beginning") is None
    assert checker.should_stop_string("the end") == "end"
