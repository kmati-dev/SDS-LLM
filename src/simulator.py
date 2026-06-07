from typing import Any, Optional, Dict, List
from src.interfaces import AbstractDrafter, AbstractVerifier, AbstractPlayback


class NGramDrafter(AbstractDrafter):
    """
    Concrete implementation of AbstractDrafter utilizing an n-gram model.
    Looks up prefix matches in a provided text corpus (tokens) and speculates next tokens.
    """

    def __init__(self, corpus_tokens: List[int], n: int = 3, draft_size: int = 3) -> None:
        self.corpus_tokens = corpus_tokens
        self.n = n
        self.draft_size = draft_size
        self.last_n_used: int = 0          # which n-gram size was used in last call
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
    ) -> Dict[str, Any]:
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


class PlaybackMetrics:
    """
    Metrics class for collecting stats during speculative decoding simulation.
    """

    def __init__(self) -> None:
        self.total_tokens_generated = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        self.max_accepted_in_single_step = 0
        self.speculative_steps = 0
        self.normal_steps = 0

        # Step-type breakdown
        self.step_types: Dict[str, int] = {
            "no_draft":    0,  # drafter returned nothing
            "full_reject": 0,  # draft produced but first token wrong
            "partial":     0,  # some tokens accepted, then mismatch
            "full_accept": 0,  # all draft tokens accepted
        }

        # Per-step accepted counts (for acceptance distribution histogram)
        self.step_accepted_counts: List[int] = []

        # Mismatch log (capped at 200 entries)
        self.mismatch_log: List[Dict] = []

        # N-gram size usage counter
        self.n_gram_usage: Dict[int, int] = {}

    def record_step(
        self,
        accepted_count: int,
        rejected_count: int,
        draft_size: int = 0,
        step_idx: int = 0,
        context_ids: Optional[List[int]] = None,
        draft_ids: Optional[List[int]] = None,
        complete_tokens: Optional[List[int]] = None,
        n_used: int = 0,
    ) -> None:
        self.speculative_steps += 1
        self.accepted_tokens += accepted_count
        self.rejected_tokens += rejected_count
        if accepted_count > self.max_accepted_in_single_step:
            self.max_accepted_in_single_step = accepted_count

        # Classify step type
        if draft_size == 0:
            self.step_types["no_draft"] += 1
        elif accepted_count == 0:
            self.step_types["full_reject"] += 1
        elif accepted_count < draft_size:
            self.step_types["partial"] += 1
        else:
            self.step_types["full_accept"] += 1

        # Track per-step accepted count
        self.step_accepted_counts.append(accepted_count)

        # Track n-gram usage
        if n_used > 0:
            self.n_gram_usage[n_used] = self.n_gram_usage.get(n_used, 0) + 1

        # Log mismatch (only when there's a real mismatch and we have full context)
        has_mismatch = draft_size > 0 and accepted_count < draft_size
        has_context = context_ids is not None and draft_ids is not None and complete_tokens is not None
        if has_mismatch and has_context and len(self.mismatch_log) < 200:
            mismatch_pos = len(context_ids) + accepted_count
            self.mismatch_log.append({
                "step": step_idx,
                "context_ids": list(context_ids[-6:]),
                "draft_ids": list(draft_ids),
                "accepted_count": accepted_count,
                "expected_id": complete_tokens[mismatch_pos] if mismatch_pos < len(complete_tokens) else None,
                "drafted_id": draft_ids[accepted_count] if accepted_count < len(draft_ids) else None,
                "n_used": n_used,
            })

    @property
    def average_accepted_per_step(self) -> float:
        if self.speculative_steps == 0:
            return 0.0
        return self.accepted_tokens / self.speculative_steps

    @property
    def speedup_ratio(self) -> float:
        if self.speculative_steps == 0:
            return 1.0
        return self.normal_steps / self.speculative_steps

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "max_accepted_in_single_step": self.max_accepted_in_single_step,
            "average_accepted_per_step": round(self.average_accepted_per_step, 2),
            "speculative_steps": self.speculative_steps,
            "normal_steps": self.normal_steps,
            "speedup_ratio": round(self.speedup_ratio, 2),
            "step_types": dict(self.step_types),
            "n_gram_usage": dict(self.n_gram_usage),
        }


class SpeculativePlayback(AbstractPlayback):
    """
    Concrete implementation of AbstractPlayback that runs token-by-token
    speculative decoding simulations against ground truth.
    """

    def __init__(
        self,
        tokenizer: Any,
        drafter: AbstractDrafter,
        verifier: AbstractVerifier,
        metrics: Optional[PlaybackMetrics] = None,
    ) -> None:
        super().__init__(tokenizer, drafter, verifier, metrics)
        self.metrics: Optional[PlaybackMetrics] = metrics

    def run_playback(self, input_data: str, use_drafter: bool = True) -> str:
        complete_tokens = self.tokenizer.encode(input_data)
        if not complete_tokens:
            return ""

        if self.metrics:
            self.metrics.normal_steps = len(complete_tokens) - 1

        current_prefix = [complete_tokens[0]]
        step_idx = 0

        while len(current_prefix) < len(complete_tokens):
            if use_drafter and self.drafter is not None:
                prefix_snapshot = list(current_prefix)
                draft_tokens = self.drafter.generate_draft(current_prefix)
                n_used = getattr(self.drafter, "last_n_used", 0)

                verification_result = self.verifier.verify(
                    draft_tokens, current_prefix, complete_tokens
                )
                accepted_tokens = verification_result["accepted_tokens"]
                accepted_count = verification_result["accepted_count"]
                rejected_count = verification_result["rejected_count"]

                if not accepted_tokens:
                    next_gt_idx = len(current_prefix)
                    if next_gt_idx < len(complete_tokens):
                        current_prefix.append(complete_tokens[next_gt_idx])
                    if self.metrics:
                        self.metrics.record_step(
                            0, 0,
                            draft_size=len(draft_tokens),
                            step_idx=step_idx,
                            context_ids=prefix_snapshot,
                            draft_ids=draft_tokens,
                            complete_tokens=complete_tokens,
                            n_used=n_used,
                        )
                else:
                    current_prefix.extend(accepted_tokens)
                    if self.metrics:
                        self.metrics.record_step(
                            accepted_count, rejected_count,
                            draft_size=len(draft_tokens),
                            step_idx=step_idx,
                            context_ids=prefix_snapshot,
                            draft_ids=draft_tokens,
                            complete_tokens=complete_tokens,
                            n_used=n_used,
                        )
            else:
                next_gt_idx = len(current_prefix)
                if next_gt_idx < len(complete_tokens):
                    current_prefix.append(complete_tokens[next_gt_idx])
                if self.metrics:
                    self.metrics.speculative_steps += 1

            step_idx += 1

        decoded_string = self.tokenizer.decode(current_prefix)
        if self.metrics:
            self.metrics.total_tokens_generated = len(current_prefix)

        return decoded_string
