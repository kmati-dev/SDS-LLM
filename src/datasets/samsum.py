"""
SAMSum dataset loader (replaces legacy Multi-News which requires an unsupported dataset script).
Corpus: dialogue/conversation between people.
Target: human-written summary of the conversation.
Semi-extractive — summaries borrow key phrases but also paraphrase and compress.
"""

from datasets import load_dataset

_CACHE = None


def _get_dataset():
    global _CACHE
    if _CACHE is None:
        print("Fetching SAMSum dataset from Hugging Face...")
        _CACHE = load_dataset("knkarthick/samsum", split="train")
    return _CACHE


def load(index: int = 0) -> tuple:
    dataset = _get_dataset()
    sample = dataset[index]
    corpus = sample["dialogue"]
    target = sample["summary"]
    return corpus, target
