"""N-gram based drafter implementation for speculative decoding."""

from typing import List

from specdecode.interface.abstractDrafter import AbstractDrafter


class NGramDrafter(AbstractDrafter):
    """
    Concrete implementation of AbstractDrafter utilising an n-gram model.

    Looks up prefix matches in a provided text corpus (tokens) and speculates
    next tokens. Backs off from the (n-1)-gram down to a 1-gram when no match
    is found at the higher order.
    """

    def __init__(self, corpus_tokens: List[int], n: int = 3, draft_size: int = 3) -> None:
        self.corpus_tokens = corpus_tokens
        self.n = n
        self.draft_size = draft_size
        self.last_n_used: int = 0  # which n-gram size was used in last call
        self.last_match_corpus_idx: int = -1  # where in corpus the match was found

    def generate_draft(self, prompt: List[int]) -> List[int]:
        """
        Generate speculative token guesses using historical n-gram matches.

        Backs off from (n-1)-gram down to a 1-gram if no matches are found.
        """
        self.last_n_used = 0
        self.last_match_corpus_idx = -1

        if not prompt or not self.corpus_tokens:
            return []

        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n:
                continue

            search_prefix = prompt[-current_n:]
            prefix_len = len(search_prefix)

            for i in range(len(self.corpus_tokens) - prefix_len - 1):
                if self.corpus_tokens[i : i + prefix_len] == search_prefix:
                    draft = self.corpus_tokens[i + prefix_len : i + prefix_len + self.draft_size]
                    if draft:
                        self.last_n_used = current_n
                        self.last_match_corpus_idx = i
                        return draft

        return []
