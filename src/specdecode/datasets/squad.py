"""
SQuAD dataset loader.
Corpus: passage (paragraph context).
Target: the answer string extracted from the passage.
SQuAD is extractive — the answer literally appears inside the passage,
so the n-gram drafter should achieve a high acceptance rate.
"""

from datasets import load_dataset

_CACHE = None


def _get_dataset():
    global _CACHE
    if _CACHE is None:
        print("Fetching SQuAD dataset from Hugging Face...")
        _CACHE = load_dataset("rajpurkar/squad", split="train")
    return _CACHE


def load(index: int = 0) -> tuple:
    dataset = _get_dataset()
    sample = dataset[index]
    corpus = sample["context"]
    target = sample["answers"]["text"][0]
    return corpus, target
