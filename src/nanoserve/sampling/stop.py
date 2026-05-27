"""Stop string detection that handles strings spanning token boundaries."""

from __future__ import annotations


class StopChecker:
    """Detect stop strings in incrementally generated text.

    Handles the subtle case where a stop string spans two consecutive tokens:
    e.g., stop="world" but tokens decode as ["wor", "ld"].
    """

    def __init__(self, stop_strings: list[str], stop_token_ids: list[int]) -> None:
        self.stop_strings = stop_strings
        self.stop_token_ids = set(stop_token_ids)

    def should_stop_token(self, token_id: int) -> bool:
        """Check if a token ID is a stop token."""
        return token_id in self.stop_token_ids

    def should_stop_string(self, generated_text: str) -> str | None:
        """Check if any stop string appears at the end of the generated text.

        Returns the matched stop string, or None.
        """
        for stop in self.stop_strings:
            if not stop:
                continue
            if generated_text.endswith(stop):
                return stop
        return None

    def check_partial_match(self, generated_text: str) -> bool:
        """Check if the generated text ends with a partial match of any stop string.

        Used to avoid outputting tokens that might be part of a stop string
        until we know for certain whether the stop string is complete.
        """
        for stop in self.stop_strings:
            if not stop:
                continue
            # Check if any suffix of generated_text is a prefix of the stop string
            for suffix_len in range(1, len(stop)):
                if generated_text.endswith(stop[:suffix_len]):
                    return True
        return False

    @property
    def has_stop_criteria(self) -> bool:
        """Whether any stop criteria are configured."""
        return bool(self.stop_strings) or bool(self.stop_token_ids)
