from typing import Any, Optional, Dict, List

import torch

from specdecode.interfaces import (
    AbstractDrafter,
    AbstractVerifier,
    AbstractPlayback,
    AbstractTensorDrafter,
    AbstractTensorVerifier,
)

# Sentinel used to right-pad candidate rows that are shorter than the draft depth.
PAD_ID = -1


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


class TensorNGramDrafter(AbstractTensorDrafter):
    """
    Tensor-emitting n-gram drafter that supports both depth- and width-drafting.

    Like NGramDrafter it matches the last (n-1) tokens of the prompt against a corpus
    (with backoff down to a 1-gram), but instead of returning a single Python list it
    returns a 2D ``torch.long`` tensor of shape ``[S, T]``:
        - depth-draft:  num_sequences=1            -> one long bet, shape [1, draft_depth]
        - width-draft:  num_sequences=S (S>1)      -> S short bets,  shape [S, draft_depth]

    Width-drafting collects several *distinct* continuations of the same matched prefix
    and prefers candidates with different first tokens, which is what hedges against an
    uncertain branch point. Rows shorter than ``draft_depth`` are right-padded with PAD_ID.
    """

    def __init__(
        self,
        corpus_tokens: List[int],
        n: int = 3,
        num_sequences: int = 1,
        draft_depth: int = 3,
    ) -> None:
        self.corpus_tokens = corpus_tokens
        self.n = n
        self.num_sequences = num_sequences      # S — number of candidate sequences
        self.draft_depth = draft_depth          # T — token depth per candidate
        self.last_n_used: int = 0               # which n-gram size was used in last call
        self.last_match_corpus_idx: int = -1    # where in corpus the first match was found

    def generate_draft(self, prompt: List[int]) -> torch.Tensor:
        self.last_n_used = 0
        self.last_match_corpus_idx = -1

        if not prompt or not self.corpus_tokens:
            return torch.empty((0, 0), dtype=torch.long)

        candidates = self._collect_candidates(prompt)
        if not candidates:
            return torch.empty((0, 0), dtype=torch.long)

        # Right-pad each candidate to draft_depth and stack into a [S, T] tensor.
        padded = [
            cand + [PAD_ID] * (self.draft_depth - len(cand))
            for cand in candidates
        ]
        return torch.tensor(padded, dtype=torch.long)

    def _collect_candidates(self, prompt: List[int]) -> List[List[int]]:
        """
        Gather up to ``num_sequences`` distinct continuations (each up to ``draft_depth``
        tokens) for the longest matching prefix, backing off from (n-1)-gram to 1-gram.
        Within the first n-size that yields any match we keep collecting, preferring
        continuations whose first token has not been seen yet.
        """
        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n:
                continue

            search_prefix = prompt[-current_n:]
            prefix_len = len(search_prefix)

            candidates: List[List[int]] = []
            seen_continuations = set()      # dedupe identical continuations
            seen_first_tokens = set()       # encourage diverse first tokens (branching)

            for i in range(len(self.corpus_tokens) - prefix_len):
                if self.corpus_tokens[i : i + prefix_len] != search_prefix:
                    continue

                draft = self.corpus_tokens[
                    i + prefix_len : i + prefix_len + self.draft_depth
                ]
                if not draft:
                    continue

                key = tuple(draft)
                if key in seen_continuations:
                    continue
                # For width-drafting, skip continuations whose first token we already
                # have, so the candidate set hedges across different branches first.
                if self.num_sequences > 1 and draft[0] in seen_first_tokens:
                    continue

                if self.last_match_corpus_idx == -1:
                    self.last_n_used = current_n
                    self.last_match_corpus_idx = i

                seen_continuations.add(key)
                seen_first_tokens.add(draft[0])
                candidates.append(draft)

                if len(candidates) >= self.num_sequences:
                    return candidates

            if candidates:
                return candidates

        return []


class TensorGreedyVerifier(AbstractTensorVerifier):
    """
    Greedy verifier for a batch of candidate draft sequences (shape ``[S, T]``).

    Every candidate row is compared token-by-token against the ground-truth sequence,
    stopping at the first mismatch, PAD_ID, or end of ground truth. The candidate with
    the longest matching run wins (ties broken by lowest row index); its matched prefix
    plus one recovery token is accepted, mirroring GreedyVerifier's single-sequence logic.
    """

    def verify(
        self,
        draft_tokens: torch.Tensor,
        current_prefix: List[int],
        complete_tokens: List[int],
    ) -> Dict[str, Any]:
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


class NGramIndex:
    """
    Pre-built ``k-gram -> [corpus positions]`` index over a token corpus, for every
    k in ``1..max_k``. Lets a drafter look up, in O(1), all positions whose preceding
    k tokens equal a query k-gram — so n-gram drafting scales to million-token corpora
    instead of the O(corpus) linear scan in NGramDrafter / TensorNGramDrafter.

    Positions for each key are stored in ascending (corpus) order, so a consumer can
    iterate them and stop early (depth needs only the first; width needs the first few).
    ``cap_positions`` bounds memory/iteration for very frequent grams (the earliest
    occurrences are kept, which is all depth/width drafting ever needs). A ``size_limit``
    at query time restricts matches to the first N corpus tokens (for corpus-size sweeps).
    """

    def __init__(self, corpus_tokens: List[int], max_k: int = 2, cap_positions: int = 256) -> None:
        self.corpus_tokens = corpus_tokens
        self.max_k = max_k
        self.cap_positions = cap_positions
        self.tables: Dict[int, Dict[tuple, List[int]]] = {}
        n_corpus = len(corpus_tokens)
        for k in range(1, max_k + 1):
            table: Dict[tuple, List[int]] = {}
            ct = corpus_tokens
            cap = cap_positions
            for i in range(n_corpus - k):
                key = tuple(ct[i:i + k])
                bucket = table.get(key)
                if bucket is None:
                    table[key] = [i]
                elif len(bucket) < cap:
                    bucket.append(i)
            self.tables[k] = table


class IndexedTensorNGramDrafter(AbstractTensorDrafter):
    """
    Index-backed equivalent of TensorNGramDrafter (depth + width drafting) that scales
    to large corpora by consulting an NGramIndex instead of scanning the corpus.

    Semantics mirror TensorNGramDrafter:
      - backoff from the (n-1)-gram down to a 1-gram,
      - within the first n-size that yields a match, collect up to ``num_sequences``
        distinct continuations (depth: S=1; width: S>1 preferring different first tokens),
      - right-pad short rows with PAD_ID, return a ``[S, T]`` long tensor (``[0,0]`` if none).

    ``size_limit`` caps matches/continuations to the first N corpus tokens, so the same
    index can drive a corpus-size sweep (0.25M, 0.5M, ... full) without rebuilding.
    """

    def __init__(
        self,
        index: NGramIndex,
        n: int = 3,
        num_sequences: int = 1,
        draft_depth: int = 3,
        size_limit: Optional[int] = None,
    ) -> None:
        self.index = index
        self.corpus_tokens = index.corpus_tokens
        self.n = n
        self.num_sequences = num_sequences
        self.draft_depth = draft_depth
        self.size_limit = size_limit if size_limit is not None else len(self.corpus_tokens)
        self.last_n_used: int = 0
        self.last_match_corpus_idx: int = -1

    def generate_draft(self, prompt: List[int]) -> torch.Tensor:
        self.last_n_used = 0
        self.last_match_corpus_idx = -1

        if not prompt:
            return torch.empty((0, 0), dtype=torch.long)

        candidates = self._collect_candidates(prompt)
        if not candidates:
            return torch.empty((0, 0), dtype=torch.long)

        padded = [
            cand + [PAD_ID] * (self.draft_depth - len(cand))
            for cand in candidates
        ]
        return torch.tensor(padded, dtype=torch.long)

    def _collect_candidates(self, prompt: List[int]) -> List[List[int]]:
        ct = self.corpus_tokens
        depth = self.draft_depth
        lim = self.size_limit

        for current_n in range(self.n - 1, 0, -1):
            if len(prompt) < current_n or current_n > self.index.max_k:
                continue

            key = tuple(prompt[-current_n:])
            positions = self.index.tables[current_n].get(key)
            if not positions:
                continue

            candidates: List[List[int]] = []
            seen_continuations = set()
            seen_first_tokens = set()

            for i in positions:
                if i >= lim:
                    break  # positions are ascending; the rest are outside the corpus slice
                start = i + current_n
                end = min(start + depth, lim)
                draft = ct[start:end]
                if not draft:
                    continue

                key_cont = tuple(draft)
                if key_cont in seen_continuations:
                    continue
                if self.num_sequences > 1 and draft[0] in seen_first_tokens:
                    continue

                if self.last_match_corpus_idx == -1:
                    self.last_n_used = current_n
                    self.last_match_corpus_idx = i

                seen_continuations.add(key_cont)
                seen_first_tokens.add(draft[0])
                candidates.append(list(draft))

                if len(candidates) >= self.num_sequences:
                    return candidates

            if candidates:
                return candidates

        return []


class TensorSpeculativePlayback(AbstractPlayback):
    """
    Playback loop driven by a tensor drafter (``[S, T]``) + tensor verifier, the missing
    piece that wires TensorNGramDrafter / TensorGreedyVerifier into an end-to-end run.

    Mirrors SpeculativePlayback but the verifier returns the best candidate row; metrics
    reuse PlaybackMetrics with ``draft_size`` = real (non-PAD) length of the chosen row,
    so step-type classification (no_draft / full_reject / partial / full_accept) and the
    mismatch log behave exactly as in the single-sequence path.
    """

    def __init__(
        self,
        tokenizer: Any,
        drafter: AbstractTensorDrafter,
        verifier: AbstractTensorVerifier,
        metrics: Optional[PlaybackMetrics] = None,
    ) -> None:
        super().__init__(tokenizer, drafter, verifier, metrics)
        self.metrics: Optional[PlaybackMetrics] = metrics

    def run_playback(self, input_data: str, use_drafter: bool = True) -> str:
        complete_tokens = self.tokenizer.encode(input_data)
        reconstructed = self.run_tokens(complete_tokens, use_drafter=use_drafter)
        return self.tokenizer.decode(reconstructed)

    def run_tokens(self, complete_tokens: List[int], use_drafter: bool = True) -> List[int]:
        """Token-level driver (skips encode/decode) — used by the analysis harness."""
        if not complete_tokens:
            return []

        if self.metrics:
            self.metrics.normal_steps = len(complete_tokens) - 1

        current_prefix = [complete_tokens[0]]
        step_idx = 0

        while len(current_prefix) < len(complete_tokens):
            if use_drafter and self.drafter is not None:
                prefix_snapshot = list(current_prefix)
                draft_tokens = self.drafter.generate_draft(current_prefix)
                n_used = getattr(self.drafter, "last_n_used", 0)

                result = self.verifier.verify(draft_tokens, current_prefix, complete_tokens)
                accepted_tokens = result["accepted_tokens"]
                accepted_count = result["accepted_count"]
                rejected_count = result["rejected_count"]
                chosen = result.get("chosen_sequence", -1)
                real_len = accepted_count + rejected_count  # non-PAD length of chosen row

                chosen_ids: List[int] = []
                if self.metrics and chosen >= 0 and draft_tokens.numel() > 0:
                    chosen_ids = [int(t) for t in draft_tokens[chosen].tolist() if t != PAD_ID]

                current_prefix.extend(accepted_tokens)
                if self.metrics:
                    self.metrics.record_step(
                        accepted_count, rejected_count,
                        draft_size=real_len,
                        step_idx=step_idx,
                        context_ids=prefix_snapshot,
                        draft_ids=chosen_ids,
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

        if self.metrics:
            self.metrics.total_tokens_generated = len(current_prefix)

        return current_prefix
