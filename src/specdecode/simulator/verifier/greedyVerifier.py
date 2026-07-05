"""Greedy verifier implementation for speculative decoding."""
from typing import Dict, List

from specdecode.interface.abstractVerifier import AbstractVerifier


class GreedyVerifier(AbstractVerifier):
    """
    Concrete implementation of AbstractVerifier that checks speculative tokens against
    the ground truth complete sequence in a greedy (sequential match) manner.
    """

    def verify(
        self,
        draft_tokens: List[int],
        current_prefix: List[int],
        complete_tokens: List[int],
    ) -> Dict[str, object]:
        prefix_len = len(current_prefix)
        accepted_draft_tokens: List[int] = []
        rejected_count = 0

        for i, token in enumerate(draft_tokens):
            gt_index = prefix_len + i
            if gt_index < len(complete_tokens) and token == complete_tokens[gt_index]:
                accepted_draft_tokens.append(token)
            else:
                rejected_count = len(draft_tokens) - i
                break

        recovery_index = prefix_len + len(accepted_draft_tokens)
        accepted_tokens = list(accepted_draft_tokens)
        if recovery_index < len(complete_tokens):
            recovery_token = complete_tokens[recovery_index]
            accepted_tokens.append(recovery_token)

        return {
            "accepted_tokens": accepted_tokens,
            "accepted_count": len(accepted_draft_tokens),
            "rejected_count": rejected_count,
        }
