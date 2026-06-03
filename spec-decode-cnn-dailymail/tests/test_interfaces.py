"""
Tests for spec-decode-cnn-dailymail

ครอบคลุม 4 ส่วนหลัก:
    1. Abstract class instantiation prevention
    2. NGramDrafter logic (รวม edge cases)
    3. GreedyVerifier logic (รวม edge cases)
    4. PlaybackMetrics calculations
    5. SpeculativePlayback end-to-end (รวม edge cases)
    6. run.py helpers: dataset caching, pair correctness, role mapping
"""

import pytest
import importlib
import sys
from unittest.mock import patch, MagicMock
from typing import Any, Dict, List, Optional

from src.interfaces import AbstractDrafter, AbstractVerifier, AbstractPlayback
from src.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback


# =============================================================================
# Shared Fixtures
# =============================================================================

class MockTokenizer:
    """Minimal tokenizer that maps each character to its ASCII code."""

    def encode(self, text: str) -> List[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(t) for t in tokens)


@pytest.fixture
def tokenizer():
    return MockTokenizer()


# =============================================================================
# 1. Abstract Class Instantiation Prevention
# =============================================================================

def test_cannot_instantiate_abstract_drafter():
    with pytest.raises(TypeError) as exc:
        AbstractDrafter()  # type: ignore
    assert "Can't instantiate" in str(exc.value) or "abstract" in str(exc.value).lower()


def test_cannot_instantiate_abstract_verifier():
    with pytest.raises(TypeError) as exc:
        AbstractVerifier()  # type: ignore
    assert "Can't instantiate" in str(exc.value) or "abstract" in str(exc.value).lower()


def test_cannot_instantiate_abstract_playback():
    with pytest.raises(TypeError) as exc:
        AbstractPlayback(
            tokenizer="dummy",
            drafter=None,    # type: ignore
            verifier=None,   # type: ignore
            metrics=None,
        )
    assert "Can't instantiate" in str(exc.value) or "abstract" in str(exc.value).lower()


# =============================================================================
# 2. NGramDrafter
# =============================================================================

def test_ngram_drafter_perfect_match():
    """(n-1)-gram prefix found → draft the next tokens."""
    corpus  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    # prompt ends with [3, 4] → should predict [5, 6, 7]
    assert drafter.generate_draft([99, 100, 3, 4]) == [5, 6, 7]


def test_ngram_drafter_backoff():
    """Falls back to smaller n-gram when large prefix not found."""
    corpus  = [1, 2, 4, 5, 6, 7]   # [3, 4] not present, but [4] is
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=2)
    assert drafter.generate_draft([3, 4]) == [5, 6]


def test_ngram_drafter_no_match():
    """Returns [] when no prefix match exists at any backoff level."""
    corpus  = [1, 2, 3]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=3)
    assert drafter.generate_draft([99, 100]) == []


def test_ngram_drafter_empty_corpus():
    """Empty corpus always returns []."""
    drafter = NGramDrafter(corpus_tokens=[], n=3, draft_size=3)
    assert drafter.generate_draft([1, 2, 3]) == []


def test_ngram_drafter_empty_prompt():
    """Empty prompt always returns []."""
    drafter = NGramDrafter(corpus_tokens=[1, 2, 3, 4, 5], n=3, draft_size=3)
    assert drafter.generate_draft([]) == []


def test_ngram_drafter_prompt_shorter_than_n():
    """Prompt shorter than n should still back off and find a 1-gram match."""
    corpus  = [1, 2, 3, 4]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=1)
    # prompt=[2], backs off to 1-gram [2] → found at index 1, next token = 3
    assert drafter.generate_draft([2]) == [3]


def test_ngram_drafter_draft_size_exceeds_remaining_corpus():
    """Returns only the tokens available when draft_size > remaining corpus."""
    corpus  = [1, 2, 3, 4, 5]
    drafter = NGramDrafter(corpus_tokens=corpus, n=2, draft_size=100)
    result  = drafter.generate_draft([1, 2, 3])
    assert len(result) <= 2   # at most 2 tokens remain after [3]


def test_ngram_drafter_boundary_condition():
    """Verify that matches at the very end of the corpus (that have continuation) are matched correctly."""
    # Corpus: [1, 2, 3, 4]
    # Prompt ending with [2, 3] should match [2, 3] at index 1 and speculate [4]
    corpus = [1, 2, 3, 4]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=1)
    result = drafter.generate_draft([2, 3])
    assert result == [4]


def test_ngram_drafter_strategies():
    """Verify frequency, recency, and first matching strategies in NGramDrafter."""
    # [9, 9, 1] (index 0)
    # [9, 9, 2] (index 3)
    # [9, 9, 2] (index 6)
    corpus_strat = [9, 9, 1, 9, 9, 2, 9, 9, 2]
    
    # first strategy -> [1]
    drafter_first = NGramDrafter(corpus_tokens=corpus_strat, n=3, draft_size=1, matching_strategy="first")
    assert drafter_first.generate_draft([9, 9]) == [1]
    
    # recency strategy -> [2]
    drafter_recency = NGramDrafter(corpus_tokens=corpus_strat, n=3, draft_size=1, matching_strategy="recency")
    assert drafter_recency.generate_draft([9, 9]) == [2]
    
    # frequency strategy -> [2] (since [2] appears twice, [1] appears once)
    drafter_freq = NGramDrafter(corpus_tokens=corpus_strat, n=3, draft_size=1, matching_strategy="frequency")
    assert drafter_freq.generate_draft([9, 9]) == [2]


def test_ngram_drafter_explain_draft():
    """Verify explain_draft returns a valid explanation dictionary."""
    corpus = [9, 9, 1, 9, 9, 2, 9, 9, 2]
    drafter = NGramDrafter(corpus_tokens=corpus, n=3, draft_size=1, matching_strategy="frequency")
    explanation = drafter.explain_draft([9, 9])
    assert "n_used" in explanation
    assert "reason" in explanation
    assert explanation["chosen_draft"] == [2]


# =============================================================================
# 3. GreedyVerifier
# =============================================================================

def test_greedy_verifier_all_accepted():
    """All draft tokens match → accepted_count == len(draft), +1 recovery token."""
    verifier = GreedyVerifier()
    result   = verifier.verify(
        draft_tokens=[4, 5, 6],
        current_prefix=[1, 2, 3],
        complete_tokens=[1, 2, 3, 4, 5, 6, 7],
    )
    assert result["accepted_count"]  == 3
    assert result["rejected_count"]  == 0
    assert result["accepted_tokens"] == [4, 5, 6, 7]   # draft + recovery


def test_greedy_verifier_partial_accepted():
    """First mismatch stops acceptance; remaining tokens are rejected."""
    verifier = GreedyVerifier()
    result   = verifier.verify(
        draft_tokens=[4, 5, 99, 100],   # 99 wrong (correct is 6)
        current_prefix=[1, 2, 3],
        complete_tokens=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    assert result["accepted_count"]  == 2
    assert result["rejected_count"]  == 2
    assert result["accepted_tokens"] == [4, 5, 6]      # [4,5] + recovery [6]


def test_greedy_verifier_all_rejected():
    """First token wrong → accepted_count == 0, only recovery token returned."""
    verifier = GreedyVerifier()
    result   = verifier.verify(
        draft_tokens=[99, 100],
        current_prefix=[1, 2, 3],
        complete_tokens=[1, 2, 3, 4, 5, 6],
    )
    assert result["accepted_count"]  == 0
    assert result["rejected_count"]  == 2
    assert result["accepted_tokens"] == [4]             # recovery only


def test_greedy_verifier_empty_draft():
    """Empty draft → only recovery token is returned."""
    verifier = GreedyVerifier()
    result   = verifier.verify(
        draft_tokens=[],
        current_prefix=[1, 2, 3],
        complete_tokens=[1, 2, 3, 4, 5],
    )
    assert result["accepted_count"]  == 0
    assert result["rejected_count"]  == 0
    assert result["accepted_tokens"] == [4]


def test_greedy_verifier_draft_exceeds_sequence_end():
    """Draft running past the end of complete_tokens stops at boundary."""
    verifier = GreedyVerifier()
    result   = verifier.verify(
        draft_tokens=[4, 5, 6, 99, 100],
        current_prefix=[1, 2, 3],
        complete_tokens=[1, 2, 3, 4, 5, 6],
    )
    assert result["accepted_count"]  == 3
    # No recovery token because we reached the very end
    assert result["accepted_tokens"] == [4, 5, 6]


# =============================================================================
# 4. PlaybackMetrics
# =============================================================================

def test_playback_metrics_initial_state():
    """Freshly created metrics must be all-zero / neutral."""
    summary = PlaybackMetrics().get_summary()
    assert summary["accepted_tokens"]            == 0
    assert summary["rejected_tokens"]            == 0
    assert summary["speculative_steps"]          == 0
    assert summary["average_accepted_per_step"]  == 0.0
    assert summary["speedup_ratio"]              == 1.0


def test_playback_metrics_calculations():
    """Multi-step recording produces correct aggregates."""
    metrics = PlaybackMetrics()
    metrics.normal_steps = 10
    metrics.record_step(accepted_count=3, rejected_count=1)
    metrics.record_step(accepted_count=1, rejected_count=3)

    summary = metrics.get_summary()
    assert summary["accepted_tokens"]           == 4
    assert summary["rejected_tokens"]           == 4
    assert summary["max_accepted_in_single_step"] == 3
    assert summary["speculative_steps"]         == 2
    assert summary["average_accepted_per_step"] == 2.0
    assert summary["speedup_ratio"]             == 5.0   # 10 / 2


def test_playback_metrics_max_accepted_tracked_correctly():
    """max_accepted_in_single_step always reflects the highest single-step count."""
    metrics = PlaybackMetrics()
    metrics.record_step(1, 2)
    metrics.record_step(5, 0)
    metrics.record_step(3, 1)
    assert metrics.get_summary()["max_accepted_in_single_step"] == 5


# =============================================================================
# 5. SpeculativePlayback End-to-End
# =============================================================================

def test_end_to_end_speculative_playback_with_drafter(tokenizer):
    """Speculative playback with a perfect corpus reconstructs the original text."""
    text   = "hello world speculative decoding"
    tokens = tokenizer.encode(text)

    playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=tokens, n=3, draft_size=3),
        verifier=GreedyVerifier(),
        metrics=(m := PlaybackMetrics()),
    )
    assert playback.run_playback(text, use_drafter=True) == text

    summary = m.get_summary()
    assert summary["accepted_tokens"] > 0
    assert summary["speculative_steps"] < summary["normal_steps"]
    assert summary["speedup_ratio"] > 1.0


def test_end_to_end_playback_without_drafter(tokenizer):
    """Normal (non-speculative) playback reconstructs the original text."""
    text = "simple run"
    playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=[], n=3, draft_size=3),
        verifier=GreedyVerifier(),
        metrics=(m := PlaybackMetrics()),
    )
    assert playback.run_playback(text, use_drafter=False) == text
    summary = m.get_summary()
    assert summary["accepted_tokens"] == 0
    assert summary["speedup_ratio"]   == 1.0


def test_playback_empty_input(tokenizer):
    """Empty input string → returns empty string immediately."""
    playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=[], n=3, draft_size=3),
        verifier=GreedyVerifier(),
    )
    assert playback.run_playback("", use_drafter=True) == ""


def test_playback_single_token_input(tokenizer):
    """Single-character input requires no generation steps."""
    playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=[], n=3, draft_size=3),
        verifier=GreedyVerifier(),
    )
    assert playback.run_playback("A") == "A"


def test_playback_drafter_returns_nothing_still_completes(tokenizer):
    """When drafter always returns [], fallback must still reconstruct correctly."""
    text = "hello"
    playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens=[], n=3, draft_size=3),
        verifier=GreedyVerifier(),
    )
    assert playback.run_playback(text, use_drafter=True) == text


def test_playback_output_identical_with_and_without_drafter(tokenizer):
    """Speculative and normal decoding must produce identical output."""
    text   = "hello world"
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


# =============================================================================
# 6. run.py Helpers: Dataset Caching, Pair Correctness, Role Mapping
# =============================================================================

# ---------------------------------------------------------------------------
# Shared fake dataset used across all run.py tests (no network required)
# ---------------------------------------------------------------------------

FAKE_DATASET = [
    {"article": "The quick brown fox jumps over the lazy dog. " * 50,
     "highlights": "Fox jumps over dog."},
    {"article": "Scientists discover new planet outside solar system. " * 40,
     "highlights": "New planet discovered."},
    {"article": "Stock markets fall amid global economic uncertainty. " * 35,
     "highlights": "Markets fall on uncertainty."},
]


def _make_mock_dataset():
    """Return a mock that behaves like a HuggingFace Dataset."""
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = len(FAKE_DATASET)
    mock_ds.__getitem__.side_effect = lambda i: FAKE_DATASET[i]
    return mock_ds


def _fresh_run_module():
    """Force-reload run module so _DATASET_CACHE resets between tests."""
    import run as run_mod
    run_mod._DATASET_CACHE = None
    return run_mod


@pytest.fixture(autouse=False)
def patched_run(tmp_path, monkeypatch):
    """Import run.py with HuggingFace patched out; reset cache before each test."""
    # Patch load_dataset globally in the run module namespace
    mock_ds = _make_mock_dataset()
    with patch("run.load_dataset", return_value=mock_ds):
        import run as run_mod
        run_mod._DATASET_CACHE = None   # clear cache before test
        yield run_mod
        run_mod._DATASET_CACHE = None   # clean up after test


# -- 6a. Dataset is loaded only once (caching) --------------------------------

def test_dataset_loaded_only_once(patched_run):
    """_get_dataset() must call load_dataset exactly once across multiple calls."""
    with patch("run.load_dataset", return_value=_make_mock_dataset()) as mock_load:
        patched_run._DATASET_CACHE = None   # ensure clean state
        patched_run._get_dataset()
        patched_run._get_dataset()
        patched_run._get_dataset()
    assert mock_load.call_count == 1


# -- 6b. Article and highlights come from the SAME row ------------------------

def test_pair_same_row(patched_run):
    """get_cnn_dailymail_pair(index) must return article and highlights from the same row."""
    for idx in range(len(FAKE_DATASET)):
        article, highlights = patched_run.get_cnn_dailymail_pair(index=idx)
        assert article    == FAKE_DATASET[idx]["article"]
        assert highlights == FAKE_DATASET[idx]["highlights"]


# -- 6c. Role mapping: article → corpus (long), highlights → target (short) --

def test_role_mapping_article_is_longer(patched_run):
    """The article (corpus) must always be longer than the highlights (target)."""
    for idx in range(len(FAKE_DATASET)):
        article, highlights = patched_run.get_cnn_dailymail_pair(index=idx)
        assert len(article) > len(highlights), (
            f"Row {idx}: article ({len(article)} chars) should be longer "
            f"than highlights ({len(highlights)} chars)"
        )


# -- 6d. Random index is consistent when no index is passed -------------------

def test_pair_random_index_returns_valid_row(patched_run):
    """get_cnn_dailymail_pair() with no index must still return a valid row."""
    article, highlights = patched_run.get_cnn_dailymail_pair()
    articles   = [row["article"]    for row in FAKE_DATASET]
    highlights_list = [row["highlights"] for row in FAKE_DATASET]
    assert article    in articles
    assert highlights in highlights_list


# -- 6e. Corpus tokens > target tokens after tokenization (role sanity) -------

def test_corpus_tokens_more_than_target_tokens(patched_run):
    """After tokenisation, corpus (article) should always have more tokens than target (summary)."""
    from transformers import AutoTokenizer
    # Use a tiny real tokenizer only if available; otherwise use MockTokenizer
    tokenizer = MockTokenizer()

    for idx in range(len(FAKE_DATASET)):
        article, highlights = patched_run.get_cnn_dailymail_pair(index=idx)
        corpus_tokens  = tokenizer.encode(article)
        target_tokens  = tokenizer.encode(highlights)
        assert len(corpus_tokens) > len(target_tokens), (
            f"Row {idx}: corpus ({len(corpus_tokens)}) <= target ({len(target_tokens)})"
        )
