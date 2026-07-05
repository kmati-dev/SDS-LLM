"""Abstract base class for tensor-based Draft Models in speculative decoding."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch


class AbstractTensorDrafter(ABC):
    """
    Tensor-based counterpart of AbstractDrafter for batched speculative decoding.

    Instead of a single ``List[int]`` sequence, the drafter returns a 2-D tensor of
    shape ``[S, T]`` (S candidate sequences, each of depth T). This naturally covers
    both strategies under a fixed token budget B:

    - depth-draft:  one long bet       -> S=1, T=B   -> shape [1, B]
    - width-draft:  several short bets -> S>1, T<B   -> shape [S, T] (S*T <= B)
    """

    @abstractmethod
    def generate_draft(self, prompt: List[int]) -> "torch.Tensor":
        """
        Generate speculative token guesses as a 2-D tensor.

        Args:
            prompt: List of token IDs representing the current prefix context.

        Returns:
            A ``torch.long`` tensor of shape ``[S, T]``. Rows shorter than T are
            right-padded with the PAD sentinel ``-1``. When no match is found an
            empty tensor of shape ``[0, 0]`` is returned.
        """
