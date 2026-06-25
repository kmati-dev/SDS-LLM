"""
XSum dataset loader.
Corpus: full news article (document).
Target: one-sentence extreme summary.
XSum is abstractive — the summary rarely copies verbatim from the article,
so the n-gram drafter will face more mismatches than on extractive datasets.
"""

from datasets import load_dataset

_CACHE = None


def _get_dataset():
    global _CACHE
    if _CACHE is None:
        print("Fetching XSum dataset from Hugging Face...")
        _CACHE = load_dataset("EdinburghNLP/xsum", split="train")
    return _CACHE


def load(index: int = 0) -> tuple:
    dataset = _get_dataset()
    sample = dataset[index]
    corpus = sample["document"]
    target = sample["summary"]
    return corpus, target
