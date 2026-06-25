"""Shared primitives for the dataset RCA analyzers."""

import json
import os
from typing import List

from specdecode.simulator import (
    NGramDrafter,
    GreedyVerifier,
    PlaybackMetrics,
    SpeculativePlayback,
)


def run_sim(corpus_tokens: List[int], target_text: str, tokenizer, k: int, n: int) -> PlaybackMetrics:
    """Run one greedy speculative-decoding playback and return its metrics."""
    m = PlaybackMetrics()
    SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens, n=n, draft_size=k),
        verifier=GreedyVerifier(),
        metrics=m,
    ).run_playback(target_text)
    return m


def lexical_overlap(corpus_tokens: List[int], target_tokens: List[int]) -> float:
    """Fraction of target tokens that appear anywhere in the corpus."""
    s = set(corpus_tokens)
    return sum(1 for t in target_tokens if t in s) / len(target_tokens) if target_tokens else 0.0


def token_novelty_rate(corpus_tokens: List[int], target_tokens: List[int]) -> float:
    """Fraction of target tokens that do NOT appear in the corpus (1 - overlap)."""
    s = set(corpus_tokens)
    novel = sum(1 for t in target_tokens if t not in s)
    return novel / len(target_tokens) if target_tokens else 0.0


def find_token_span(haystack: List[int], needle: List[int]) -> int:
    """First index where ``needle`` occurs contiguously in ``haystack`` (-1 if absent)."""
    if not needle:
        return -1
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
