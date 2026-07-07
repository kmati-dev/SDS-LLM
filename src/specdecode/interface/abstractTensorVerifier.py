"""Abstract base class for tensor-based Verifiers in speculative decoding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import torch


class AbstractTensorVerifier(ABC):
    """
    Tensor-based counterpart of AbstractVerifier.

    Accepts a batch of candidate draft sequences (shape ``[S, T]``), greedily
    verifies every candidate against the ground-truth sequence, and keeps only the
    single candidate whose prefix matches the ground truth for the longest run.
    """

    @abstractmethod
    def verify(
        self,
        draft_tokens: "torch.Tensor",
        current_prefix: List[int],
        complete_tokens: List[int],
    ) -> Dict[str, object]:
        """
        Greedily verify a batch of speculative draft sequences against ground truth.

        Args:
            draft_tokens: ``torch.long`` tensor of shape ``[S, T]`` (PAD = -1).
            current_prefix: Token IDs accepted/generated so far.
            complete_tokens: All ground-truth token IDs for the final sequence.

        Returns:
            {
                "accepted_tokens": List[int],  # winning candidate prefix + 1 recovery
                "accepted_count": int,         # draft tokens accepted from the winner
                "rejected_count": int,         # remaining tokens in the winner (unaccepted)
                "chosen_sequence": int,        # row index of winning candidate (-1 if none)
            }
        """
