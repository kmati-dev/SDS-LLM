"""Metrics collector for speculative decoding simulation steps."""

from typing import Dict, List, Optional


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
            "no_draft": 0,  # drafter returned nothing
            "full_reject": 0,  # draft produced but first token wrong
            "partial": 0,  # some tokens accepted, then mismatch
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
        has_context = (
            context_ids is not None and draft_ids is not None and complete_tokens is not None
        )
        if has_mismatch and has_context and len(self.mismatch_log) < 200:
            mismatch_pos = len(context_ids) + accepted_count  # type: ignore[arg-type]
            self.mismatch_log.append(
                {
                    "step": step_idx,
                    "context_ids": list(context_ids[-6:]),  # type: ignore[index]
                    "draft_ids": list(draft_ids),  # type: ignore[arg-type]
                    "accepted_count": accepted_count,
                    "expected_id": (
                        complete_tokens[mismatch_pos]  # type: ignore[index]
                        if mismatch_pos < len(complete_tokens)  # type: ignore[arg-type]
                        else None
                    ),
                    "drafted_id": (
                        draft_ids[accepted_count]  # type: ignore[index]
                        if accepted_count < len(draft_ids)  # type: ignore[arg-type]
                        else None
                    ),
                    "n_used": n_used,
                }
            )

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

    def get_summary(self) -> Dict[str, object]:
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
