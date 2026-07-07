"""Abstract base class for the Playback loop in speculative decoding."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from specdecode.interface.abstractDrafter import AbstractDrafter
from specdecode.interface.abstractVerifier import AbstractVerifier


class AbstractPlayback(ABC):
    """
    Abstract base class controlling the execution loop of the speculative decoding
    simulator.

    Coordinates the interaction between the Drafter, Verifier, Tokenizer, and Metrics
    tracker.
    """

    def __init__(
        self,
        tokenizer: Any,
        drafter: AbstractDrafter,
        verifier: AbstractVerifier,
        metrics: Optional[Any] = None,
    ) -> None:
        """
        Initialise the speculative decoding playback simulator with Dependency Injection.

        Args:
            tokenizer: Any tokenizer object (e.g. HuggingFace) that exposes
                ``encode(str) -> List[int]`` and ``decode(List[int]) -> str``.
                Duck-typed intentionally.
            drafter: An instance of a class derived from AbstractDrafter.
            verifier: An instance of a class derived from AbstractVerifier.
            metrics: Optional metrics tracker. Defaults to None (disabled).
        """
        self.tokenizer = tokenizer
        self.drafter = drafter
        self.verifier = verifier
        self.metrics = metrics

    @abstractmethod
    def run_playback(self, input_data: str, use_drafter: bool = True) -> str:
        """
        Run the token-by-token decoding playback simulation.

        Args:
            input_data: The prompt string to start the generation or verification
                process.
            use_drafter: Flag to toggle whether the Speculative Drafter should be
                utilised.

        Returns:
            The final decoded output string.
        """
