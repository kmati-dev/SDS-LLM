import pytest
from typing import Any, Optional, Dict, List

from src.interfaces import AbstractDrafter, AbstractVerifier, AbstractPlayback
from src.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback


# =============================================================================
# 1. Tests for Instantiation Prevention (Abstract Classes)
# =============================================================================

def test_cannot_instantiate_abstract_drafter():
    """Verify that AbstractDrafter cannot be instantiated directly."""
    with pytest.raises(TypeError) as excinfo:
        AbstractDrafter()  # type: ignore
    assert "Can't instantiate abstract class" in str(excinfo.value) or "Can't instantiate class" in str(excinfo.value)


def test_cannot_instantiate_abstract_verifier():
    """Verify that AbstractVerifier cannot be instantiated directly."""
    with pytest.raises(TypeError) as excinfo:
        AbstractVerifier()  # type: ignore
    assert "Can't instantiate abstract class" in str(excinfo.value) or "Can't instantiate class" in str(excinfo.value)


def test_cannot_instantiate_abstract_playback():
    """Verify that AbstractPlayback cannot be instantiated directly."""
    with pytest.raises(TypeError) as excinfo:
        AbstractPlayback(
            tokenizer="dummy_tokenizer",
            drafter=None,  # type: ignore
            verifier=None,  # type: ignore
            metrics=None
        )  # type: ignore
    assert "Can't instantiate abstract class" in str(excinfo.value) or "Can't instantiate class" in str(excinfo.value)


# =============================================================================
# 2. Tests for Concrete NGramDrafter Implementation
# =============================================================================

def test_ngram_drafter_perfect_match():
    """Verify NGramDrafter speculates correct tokens when a perfect (n-1)-gram is found in the corpus."""
    # Corpus: [1, 2, 3, 4, 5, 6, 7]
    # If N=3, search-prefix size is 2.
    # If prompt ends with [3, 4], the matched suffix in corpus should speculate [5, 6, 7]
    corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    
    draft = drafter.generate_draft([99, 100, 3, 4])
    assert draft == [5, 6, 7]


def test_ngram_drafter_backoff():
    """Verify NGramDrafter backs off to smaller n-grams if the large n-gram matches nothing."""
    # Corpus contains pattern [4, 5, 6], but NOT [3, 4, 5].
    # Prompt is [3, 4]. Under N=3, it searches for [3, 4]. No match.
    # It must back off to [4] (1-gram) and find the first match [4], speculating [5, 6]
    corpus = [1, 2, 4, 5, 6, 7]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=2)
    
    draft = drafter.generate_draft([3, 4])
    assert draft == [5, 6]


def test_ngram_drafter_no_match():
    """Verify NGramDrafter returns an empty list if absolutely no match is found."""
    corpus = [1, 2, 3]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    
    draft = drafter.generate_draft([99, 100])
    assert draft == []


# =============================================================================
# 3. Tests for Concrete GreedyVerifier Implementation
# =============================================================================

def test_greedy_verifier_all_accepted():
    """Verify GreedyVerifier behavior when all speculative tokens match the ground truth."""
    verifier = GreedyVerifier()
    draft = [4, 5, 6]
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6, 7]  # ground truth
    
    result = verifier.verify(draft, prefix, complete)
    
    assert result["accepted_count"] == 3
    assert result["rejected_count"] == 0
    # Accepted tokens = draft + 1 recovery token [7]
    assert result["accepted_tokens"] == [4, 5, 6, 7]


def test_greedy_verifier_partial_accepted():
    """Verify GreedyVerifier behavior when some speculative tokens match and then a mismatch occurs."""
    verifier = GreedyVerifier()
    draft = [4, 5, 99, 100]  # 99 is incorrect, correct is 6
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6, 7, 8]
    
    result = verifier.verify(draft, prefix, complete)
    
    assert result["accepted_count"] == 2  # [4, 5] accepted
    assert result["rejected_count"] == 2  # [99, 100] rejected
    # Accepted tokens = accepted draft tokens [4, 5] + 1 recovery token [6]
    assert result["accepted_tokens"] == [4, 5, 6]


def test_greedy_verifier_all_rejected():
    """Verify GreedyVerifier behavior when the very first speculative token is incorrect."""
    verifier = GreedyVerifier()
    draft = [99, 100]  # correct is [4, 5]
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6]
    
    result = verifier.verify(draft, prefix, complete)
    
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 2
    # Accepted tokens = only recovery token [4]
    assert result["accepted_tokens"] == [4]


# =============================================================================
# 4. Tests for PlaybackMetrics Implementation
# =============================================================================

def test_playback_metrics_calculations():
    """Verify metrics calculation logic for steps, counts, average, and ratio."""
    metrics = PlaybackMetrics()
    metrics.normal_steps = 10
    
    # Step 1: 3 accepted, 1 rejected
    metrics.record_step(accepted_count=3, rejected_count=1)
    # Step 2: 1 accepted, 3 rejected
    metrics.record_step(accepted_count=1, rejected_count=3)
    
    summary = metrics.get_summary()
    
    assert summary["accepted_tokens"] == 4
    assert summary["rejected_tokens"] == 4
    assert summary["max_accepted_in_single_step"] == 3
    assert summary["speculative_steps"] == 2
    assert summary["average_accepted_per_step"] == 2.0
    # Speedup: normal_steps (10) / speculative_steps (2) = 5.0
    assert summary["speedup_ratio"] == 5.0


# =============================================================================
# 5. Mock Tokenizer and End-to-End Playback Tests
# =============================================================================

class MockTokenizer:
    """Mock Tokenizer supporting duck typing (encode and decode)."""
    
    def encode(self, text: str) -> List[int]:
        # Converts letters to ASCII token IDs
        return [ord(char) for char in text]

    def decode(self, tokens: List[int]) -> str:
        # Converts ASCII token IDs back to a string
        return "".join(chr(t) for t in tokens)


def test_end_to_end_speculative_playback_with_drafter():
    """Verify complete playback simulator using Speculative Decoding."""
    tokenizer = MockTokenizer()
    ground_truth_text = "hello world speculative decoding"
    ground_truth_tokens = tokenizer.encode(ground_truth_text)
    
    # Let's seed the drafter with the exact ground truth tokens so it has a reference
    # database to pull perfect match guesses from!
    drafter = NGramDrafter(corpus_tokens=ground_truth_tokens, n=3, draft_size=3)
    verifier = GreedyVerifier()
    metrics = PlaybackMetrics()
    
    playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=drafter,
        verifier=verifier,
        metrics=metrics
    )
    
    reconstructed_text = playback.run_playback(ground_truth_text, use_drafter=True)
    
    # Playback must correctly rebuild the original string
    assert reconstructed_text == ground_truth_text
    
    # Verify metrics show that speculative steps are far fewer than normal steps
    summary = metrics.get_summary()
    assert summary["accepted_tokens"] > 0
    assert summary["speculative_steps"] < summary["normal_steps"]
    assert summary["speedup_ratio"] > 1.0


def test_end_to_end_playback_without_drafter():
    """Verify complete playback simulator executing step-by-step normal decoding."""
    tokenizer = MockTokenizer()
    ground_truth_text = "simple run"
    
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    verifier = GreedyVerifier()
    metrics = PlaybackMetrics()
    
    playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=drafter,
        verifier=verifier,
        metrics=metrics
    )
    
    reconstructed_text = playback.run_playback(ground_truth_text, use_drafter=False)
    
    assert reconstructed_text == ground_truth_text
    summary = metrics.get_summary()
    assert summary["accepted_tokens"] == 0
    # Without drafter, the speedup ratio is exactly 1.0
    assert summary["speedup_ratio"] == 1.0


# =============================================================================
# 6. Edge Cases for NGramDrafter
# =============================================================================

def test_ngram_drafter_empty_corpus():
    """Empty corpus must always return an empty draft."""
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    assert drafter.generate_draft([1, 2, 3]) == []


def test_ngram_drafter_empty_prompt():
    """Empty prompt must return an empty draft."""
    drafter = NGramDrafter(corpus_tokens=[1, 2, 3, 4, 5], n=3, draft_size=3)
    assert drafter.generate_draft([]) == []


def test_ngram_drafter_prompt_shorter_than_n():
    """Prompt shorter than n should backoff and still find a match."""
    # corpus has [1, 2, 3], n=3 → tries 2-gram first (skipped: prompt too short),
    # then backs off to 1-gram [2] and finds continuation [3]
    corpus = [1, 2, 3, 4]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=1)
    result = drafter.generate_draft([2])
    assert result == [3]


def test_ngram_drafter_draft_size_exceeds_remaining_corpus():
    """Draft size larger than remaining corpus should return what's available."""
    corpus = [1, 2, 3, 4, 5]
    drafter = NGramDrafter(corpus_tokens=corpus, n=2, draft_size=100)
    result = drafter.generate_draft([1, 2, 3])
    # starts from position after [3] → [4, 5], nothing after 5
    assert len(result) <= 2


# =============================================================================
# 7. Edge Cases for GreedyVerifier
# =============================================================================

def test_greedy_verifier_empty_draft():
    """Empty draft should return only the recovery token."""
    verifier = GreedyVerifier()
    result = verifier.verify(
        draft_tokens=[],
        current_prefix=[1, 2, 3],
        complete_tokens=[1, 2, 3, 4, 5],
    )
    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4]  # recovery token only


def test_greedy_verifier_draft_exceeds_sequence_end():
    """Draft that goes past complete_tokens boundary stops at boundary."""
    verifier = GreedyVerifier()
    result = verifier.verify(
        draft_tokens=[4, 5, 6, 99, 100],
        current_prefix=[1, 2, 3],
        complete_tokens=[1, 2, 3, 4, 5, 6],
    )
    # [4, 5, 6] match; [99, 100] are past the boundary → rejected
    assert result["accepted_count"] == 3
    assert result["accepted_tokens"] == [4, 5, 6]  # no recovery token (end of sequence)


# =============================================================================
# 8. Edge Cases for PlaybackMetrics
# =============================================================================

def test_playback_metrics_initial_state():
    """Freshly created metrics should report all zeros."""
    metrics = PlaybackMetrics()
    summary = metrics.get_summary()
    assert summary["accepted_tokens"] == 0
    assert summary["rejected_tokens"] == 0
    assert summary["speculative_steps"] == 0
    assert summary["average_accepted_per_step"] == 0.0
    assert summary["speedup_ratio"] == 1.0


def test_playback_metrics_max_accepted_tracked_correctly():
    """max_accepted_in_single_step reflects the highest single-step accept count."""
    metrics = PlaybackMetrics()
    metrics.record_step(accepted_count=1, rejected_count=2)
    metrics.record_step(accepted_count=5, rejected_count=0)
    metrics.record_step(accepted_count=3, rejected_count=1)
    assert metrics.get_summary()["max_accepted_in_single_step"] == 5


# =============================================================================
# 9. Edge Cases for SpeculativePlayback
# =============================================================================

def test_playback_empty_input():
    """Empty input string should return an empty string immediately."""
    tokenizer = MockTokenizer()
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    verifier = GreedyVerifier()
    playback = SpeculativePlayback(tokenizer=tokenizer, drafter=drafter, verifier=verifier)
    assert playback.run_playback("", use_drafter=True) == ""


def test_playback_single_token_input():
    """Single-token input should be returned as-is without any generation steps."""
    tokenizer = MockTokenizer()
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    verifier = GreedyVerifier()
    playback = SpeculativePlayback(tokenizer=tokenizer, drafter=drafter, verifier=verifier)
    assert playback.run_playback("A") == "A"


def test_playback_drafter_returns_nothing_still_completes():
    """When drafter always returns empty, playback falls back and still reconstructs correctly."""
    tokenizer = MockTokenizer()
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)  # empty corpus → always []
    verifier = GreedyVerifier()
    playback = SpeculativePlayback(tokenizer=tokenizer, drafter=drafter, verifier=verifier)
    text = "hello"
    assert playback.run_playback(text, use_drafter=True) == text


def test_playback_output_identical_with_and_without_drafter():
    """Speculative and normal decoding must produce the exact same output string."""
    tokenizer = MockTokenizer()
    text = "hello world"
    tokens = tokenizer.encode(text)

    result_spec = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=tokens, n=3, draft_size=3),
        verifier=GreedyVerifier(),
    ).run_playback(text, use_drafter=True)

    result_norm = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=tokens, n=3, draft_size=3),
        verifier=GreedyVerifier(),
    ).run_playback(text, use_drafter=False)

    assert result_spec == result_norm == text
