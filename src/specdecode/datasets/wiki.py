"""
Generic multilingual Wikipedia loader (Hugging Face ``wikimedia/wikipedia``).

A teammate runs the whole RCA on a *different* language by changing ONLY the
language code — e.g. ``lo`` (Lao), ``my`` (Burmese), ``ar`` (Arabic),
``ru`` (Russian), ``uk`` (Ukrainian). Everything downstream is language-agnostic.

    from specdecode.datasets.wiki import load_articles
    texts = load_articles("lo")            # all Lao articles
    texts = load_articles("ru", max_chars=50_000_000)  # cap huge languages
"""

from typing import List, Optional

from datasets import load_dataset

DEFAULT_DATE = "20231101"
_CACHE = {}


def available_config(lang: str, date: str = DEFAULT_DATE) -> str:
    """Config/subset name used by wikimedia/wikipedia, e.g. '20231101.lo'."""
    return f"{date}.{lang}"


def load_articles(
    lang: str,
    date: str = DEFAULT_DATE,
    max_chars: Optional[int] = None,
    streaming: bool = False,
) -> List[str]:
    """
    Return a list of plain-text Wikipedia articles for ``lang``.

    Args:
        lang:      ISO language code (lo, my, ar, ru, uk, ...).
        date:      Wikipedia dump snapshot (default 20231101).
        max_chars: stop once this many characters have been collected
                   (use for very large languages; None = load everything).
        streaming: stream instead of materialising the full split first
                   (recommended together with max_chars for big languages).
    """
    config = available_config(lang, date)
    cache_key = (config, max_chars, streaming)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=streaming)

    texts: List[str] = []
    total = 0
    for row in ds:
        t = row["text"]
        texts.append(t)
        total += len(t)
        if max_chars is not None and total >= max_chars:
            break

    _CACHE[cache_key] = texts
    return texts
