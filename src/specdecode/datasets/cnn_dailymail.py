"""
CNN/DailyMail dataset loader.
Corpus: full news article.
Target: the human-written highlights (multi-sentence summary).
The summary is largely abstractive but reuses many entities and phrases from the
article, so the n-gram drafter lands between extractive SQuAD and abstractive XSum.
"""

from datasets import load_dataset

_CACHE = None


def _get_dataset():
    global _CACHE
    if _CACHE is None:
        print("Fetching CNN/DailyMail dataset from Hugging Face...")
        _CACHE = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    return _CACHE


def load(index: int = 0) -> tuple:
    dataset = _get_dataset()
    sample = dataset[index]
    corpus = sample["article"]
    target = sample["highlights"]
    return corpus, target
