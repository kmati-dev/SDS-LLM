"""Tensor-based greedy verifier for batched speculative decoding."""

from typing import Dict, List

import torch

from specdecode.interface.abstractTensorVerifier import AbstractTensorVerifier
from specdecode.simulator.drafter.tensorNGramDrafter import PAD_ID


class TensorGreedyVerifier(AbstractTensorVerifier):
    """
    Greedy verifier for a batch of candidate draft sequences (shape ``[S, T]``).

    Every candidate row is compared token-by-token against the ground-truth sequence,
    stopping at the first mismatch, PAD_ID, or end of ground truth. The candidate with
    the longest matching run wins (ties broken by lowest row index); its matched prefix
    plus one recovery token is accepted, mirroring GreedyVerifier's single-sequence
    logic.
    """

    def verify(
        self,
        draft_tokens: torch.Tensor,
        current_prefix: List[int],
        complete_tokens: List[int],
    ) -> Dict[str, object]:
        prefix_len = len(current_prefix)
        complete_len = len(complete_tokens)

        best_row = -1
        best_accepted: List[int] = []

        # Treat an empty draft (no candidates / width 0) as "no draft".
        if draft_tokens.numel() > 0:
            for row_idx in range(draft_tokens.shape[0]):
                row = draft_tokens[row_idx]
                accepted: List[int] = []
                for i in range(row.shape[0]):
                    token = int(row[i])
                    if token == PAD_ID:
                        break
                    gt_index = prefix_len + i
                    if gt_index < complete_len and token == complete_tokens[gt_index]:
                        accepted.append(token)
                    else:
                        break
                # First row seeds the winner; later rows only win by a strictly longer
                # match, so ties keep the lowest row index.
                if best_row == -1 or len(accepted) > len(best_accepted):
                    best_accepted = accepted
                    best_row = row_idx

        accepted_count = len(best_accepted)
        # Rejected = remaining real (non-PAD) tokens in the winning row after the match.
        rejected_count = 0
        if best_row != -1:
            winner = draft_tokens[best_row]
            real_len = int((winner != PAD_ID).sum())
            rejected_count = real_len - accepted_count

        recovery_index = prefix_len + accepted_count
        accepted_tokens = list(best_accepted)
        if recovery_index < complete_len:
            accepted_tokens.append(complete_tokens[recovery_index])

        return {
            "accepted_tokens": accepted_tokens,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "chosen_sequence": best_row,
        }
