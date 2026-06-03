from typing import Any, Optional, Dict, List
from src.interfaces import AbstractDrafter, AbstractVerifier, AbstractPlayback


class NGramDrafter(AbstractDrafter):
    """
    Concrete implementation of AbstractDrafter utilizing an n-gram model.
    Looks up prefix matches in a provided text corpus (tokens) and speculates next tokens.
    """

    def __init__(self, corpus_tokens: List[int], n: int = 3, draft_size: int = 3, matching_strategy: str = "frequency") -> None:
        """
        Initialize the N-gram Drafter.

        Args:
            corpus_tokens: The database/corpus of token IDs to search for patterns.
            n: The size of the n-gram context (will use up to n-1 tokens for prefix lookup).
            draft_size: The number of tokens to speculate (draft size K).
            matching_strategy: The strategy to choose draft tokens if multiple matches are found.
                               Options: "frequency" (default), "recency", "first".
        """
        self.corpus_tokens = corpus_tokens
        self.n = n
        self.draft_size = draft_size
        self.matching_strategy = matching_strategy

    def generate_draft(self, prompt: List[int]) -> List[int]:
        """
        Generate speculative token guesses using historical n-gram matches.
        Backs off from (n-1)-gram down to a 1-gram if no matches are found.
        """
        if not prompt or not self.corpus_tokens:
            return []

        # Attempt to match from the largest prefix context size down to 1 token
        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n:
                continue

            # Extracted target context from the end of our current prompt
            search_prefix = prompt[-current_n:]
            prefix_len = len(search_prefix)

            # Find all matches in the corpus (fixing the boundary condition range)
            matches = []
            for i in range(len(self.corpus_tokens) - prefix_len):
                if self.corpus_tokens[i : i + prefix_len] == search_prefix:
                    matches.append(i)

            if not matches:
                continue

            # Execute matching strategy to pick the draft tokens
            if self.matching_strategy == "first":
                for idx in matches:
                    draft = self.corpus_tokens[idx + prefix_len : idx + prefix_len + self.draft_size]
                    if draft:
                        return draft
            elif self.matching_strategy == "recency":
                for idx in reversed(matches):
                    draft = self.corpus_tokens[idx + prefix_len : idx + prefix_len + self.draft_size]
                    if draft:
                        return draft
            else:  # "frequency"
                from collections import Counter
                candidates = []
                for idx in matches:
                    draft = self.corpus_tokens[idx + prefix_len : idx + prefix_len + self.draft_size]
                    if draft:
                        candidates.append((tuple(draft), idx))
                if candidates:
                    freq = Counter(c[0] for c in candidates)
                    # Sort by frequency (descending) and then index (descending for recency tiebreaker)
                    candidates.sort(key=lambda c: (freq[c[0]], c[1]), reverse=True)
                    return list(candidates[0][0])

        return []

    def explain_draft(self, prompt: List[int]) -> Dict[str, Any]:
        """
        Explain how the draft was generated for the given prompt context.
        Useful for Root Cause Analysis (RCA).
        """
        if not prompt or not self.corpus_tokens:
            return {"reason": "Empty prompt or empty corpus"}

        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n:
                continue

            search_prefix = prompt[-current_n:]
            prefix_len = len(search_prefix)

            matches = []
            for i in range(len(self.corpus_tokens) - prefix_len):
                if self.corpus_tokens[i : i + prefix_len] == search_prefix:
                    matches.append(i)

            if not matches:
                continue

            explanation = {
                "n_used": current_n + 1,
                "prefix_tokens": search_prefix,
                "total_matches": len(matches),
                "strategy": self.matching_strategy,
            }

            if self.matching_strategy == "first":
                for idx in matches:
                    draft = self.corpus_tokens[idx + prefix_len : idx + prefix_len + self.draft_size]
                    if draft:
                        explanation["chosen_match_index"] = idx
                        explanation["chosen_draft"] = draft
                        explanation["reason"] = f"First match of prefix {search_prefix} at index {idx}"
                        return explanation
            elif self.matching_strategy == "recency":
                for idx in reversed(matches):
                    draft = self.corpus_tokens[idx + prefix_len : idx + prefix_len + self.draft_size]
                    if draft:
                        explanation["chosen_match_index"] = idx
                        explanation["chosen_draft"] = draft
                        explanation["reason"] = f"Most recent (last) match of prefix {search_prefix} at index {idx}"
                        return explanation
            else:  # "frequency"
                from collections import Counter
                candidates = []
                for idx in matches:
                    draft = self.corpus_tokens[idx + prefix_len : idx + prefix_len + self.draft_size]
                    if draft:
                        candidates.append((tuple(draft), idx))
                if candidates:
                    freq = Counter(c[0] for c in candidates)
                    candidates.sort(key=lambda c: (freq[c[0]], c[1]), reverse=True)
                    chosen_draft, chosen_idx = candidates[0]
                    explanation["chosen_match_index"] = chosen_idx
                    explanation["chosen_draft"] = list(chosen_draft)
                    explanation["frequency_map"] = {str(list(k)): v for k, v in freq.items()}
                    explanation["reason"] = (
                        f"Most frequent continuation {list(chosen_draft)} (frequency {freq[chosen_draft]}) "
                        f"among {len(candidates)} candidates matching prefix {search_prefix}"
                    )
                    return explanation

        return {"reason": "No match found at any n-gram level"}


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
        """
        Verify the drafted tokens against the target ground truth.
        Compares token by token until the first mismatch.
        """
        prefix_len = len(current_prefix)
        accepted_draft_tokens: List[int] = []
        rejected_count = 0

        # Greedy match loop
        for i, token in enumerate(draft_tokens):
            gt_index = prefix_len + i
            # Check if within bounds and matching the ground truth token
            if gt_index < len(complete_tokens) and token == complete_tokens[gt_index]:
                accepted_draft_tokens.append(token)
            else:
                # Mismatch found: all subsequent tokens in this draft are rejected
                rejected_count = len(draft_tokens) - i
                break

        # Calculate the recovery token (the actual ground truth token at the first mismatch point)
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
        self.normal_steps = 0  # Steps taken if doing simple token-by-token decoding
        self.mismatch_records: List[Dict[str, Any]] = []

    def record_step(self, accepted_count: int, rejected_count: int) -> None:
        """Record the metrics for a single simulation step."""
        self.speculative_steps += 1
        self.accepted_tokens += accepted_count
        self.rejected_tokens += rejected_count
        
        if accepted_count > self.max_accepted_in_single_step:
            self.max_accepted_in_single_step = accepted_count

    @property
    def average_accepted_per_step(self) -> float:
        """Calculate the average number of accepted draft tokens per speculative step."""
        if self.speculative_steps == 0:
            return 0.0
        return self.accepted_tokens / self.speculative_steps

    @property
    def speedup_ratio(self) -> float:
        """
        Speedup ratio calculated as normal_steps / speculative_steps.
        A higher ratio represents a faster simulation rate.
        """
        if self.speculative_steps == 0:
            return 1.0
        return self.normal_steps / self.speculative_steps

    def get_summary(self) -> Dict[str, Any]:
        """Return a formatted dictionary summarizing the metrics."""
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "max_accepted_in_single_step": self.max_accepted_in_single_step,
            "average_accepted_per_step": round(self.average_accepted_per_step, 2),
            "speculative_steps": self.speculative_steps,
            "normal_steps": self.normal_steps,
            "speedup_ratio": round(self.speedup_ratio, 2),
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
        """
        Initialize the playback controller with dependencies.
        """
        super().__init__(tokenizer, drafter, verifier, metrics)
        self.metrics: Optional[PlaybackMetrics] = metrics

    def run_playback(self, input_data: str, use_drafter: bool = True) -> str:
        """
        Run the simulation. input_data represents the complete target ground truth text.
        Returns the reconstructed decoded string.
        """
        # Encode complete ground truth text to token IDs
        complete_tokens = self.tokenizer.encode(input_data)
        if not complete_tokens:
            return ""

        # Set normal steps metric to the size of tokens that need generation (excluding the starting token)
        if self.metrics:
            self.metrics.normal_steps = len(complete_tokens) - 1

        # Start with the first token of the target ground truth as current prefix
        current_prefix = [complete_tokens[0]]

        while len(current_prefix) < len(complete_tokens):
            if use_drafter and self.drafter is not None:
                # 1. Speculate tokens using Drafter
                draft_tokens = self.drafter.generate_draft(current_prefix)

                # 2. Greedily verify speculative tokens against ground truth using Verifier
                verification_result = self.verifier.verify(
                    draft_tokens, current_prefix, complete_tokens
                )

                accepted_tokens = verification_result["accepted_tokens"]
                accepted_count = verification_result["accepted_count"]
                rejected_count = verification_result["rejected_count"]

                # Track mismatches for RCA
                if use_drafter and self.drafter is not None and self.metrics is not None:
                    if not draft_tokens:
                        explanation = {}
                        if hasattr(self.drafter, "explain_draft"):
                            explanation = self.drafter.explain_draft(current_prefix)
                        mismatch_info = {
                            "step_index": self.metrics.speculative_steps + 1,
                            "prompt_context_ids": list(current_prefix),
                            "draft_token_ids": [],
                            "accepted_count": 0,
                            "mismatched_draft_token_id": None,
                            "expected_token_id": complete_tokens[len(current_prefix)] if len(current_prefix) < len(complete_tokens) else None,
                            "explanation": explanation
                        }
                        self.metrics.mismatch_records.append(mismatch_info)
                    elif accepted_count < len(draft_tokens):
                        mismatched_token_id = draft_tokens[accepted_count]
                        gt_idx = len(current_prefix) + accepted_count
                        expected_token_id = complete_tokens[gt_idx] if gt_idx < len(complete_tokens) else None
                        
                        explanation = {}
                        if hasattr(self.drafter, "explain_draft"):
                            explanation = self.drafter.explain_draft(current_prefix)
                        
                        mismatch_info = {
                            "step_index": self.metrics.speculative_steps + 1,
                            "prompt_context_ids": list(current_prefix),
                            "draft_token_ids": list(draft_tokens),
                            "accepted_count": accepted_count,
                            "mismatched_draft_token_id": mismatched_token_id,
                            "expected_token_id": expected_token_id,
                            "explanation": explanation
                        }
                        self.metrics.mismatch_records.append(mismatch_info)

                # If the drafter returned nothing or everything was rejected, 
                # verify will at least return the single recovery token.
                if not accepted_tokens:
                    # Fallback boundary check: manually add the next ground truth token
                    next_gt_idx = len(current_prefix)
                    if next_gt_idx < len(complete_tokens):
                        current_prefix.append(complete_tokens[next_gt_idx])
                    if self.metrics:
                        self.metrics.record_step(0, 0)
                else:
                    # Append the accepted tokens + recovery token
                    current_prefix.extend(accepted_tokens)
                    if self.metrics:
                        self.metrics.record_step(accepted_count, rejected_count)
            else:
                # Normal token-by-token decoding: append the single next ground truth token
                next_gt_idx = len(current_prefix)
                if next_gt_idx < len(complete_tokens):
                    current_prefix.append(complete_tokens[next_gt_idx])
                if self.metrics:
                    self.metrics.speculative_steps += 1  # Standard steps count up too for fair comparison

        # Decode token IDs back to a final string
        decoded_string = self.tokenizer.decode(current_prefix)
        if self.metrics:
            self.metrics.total_tokens_generated = len(current_prefix)

        return decoded_string
