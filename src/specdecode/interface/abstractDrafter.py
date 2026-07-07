"""Abstract base class for the Draft Model in speculative decoding."""

from abc import ABC, abstractmethod
from typing import List


class AbstractDrafter(ABC):
    """
    Abstract base class representing the Draft Model in speculative decoding.

    The Drafter generates a small sequence of speculative tokens (guesses)
    based on the current prefix sequence.
    """

    @abstractmethod
    def generate_draft(self, prompt: List[int]) -> List[int]:
        """
        Generate speculative token guesses for the next words/tokens.

        Args:
            prompt: List of token IDs representing the current prefix context.

        Returns:
            A list of guessed token IDs (draft tokens). Empty list if no
            match is found.
        """
