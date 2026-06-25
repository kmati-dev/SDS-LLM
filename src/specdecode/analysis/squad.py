"""SQuAD analyzer — extractive QA (answer is a literal span of the passage)."""

from typing import List

from datasets import load_dataset

from .common import find_token_span
from .engine import DatasetAnalyzer


class SquadAnalyzer(DatasetAnalyzer):
    name = "squad"
    title = "SQuAD (Extractive)"
    color = "#3b82f6"
    marker = "o"
    fixed_n = 3
    max_draft = 6
    alignment_label = "Lexical Overlap"

    def load_raw(self):
        return load_dataset("rajpurkar/squad", split="train")

    def extract(self, sample):
        return sample["context"], sample["answers"]["text"][0]

    def deepdive(self, tokenizer, ct: List[int], tt: List[int]) -> None:
        span = find_token_span(ct, tt)
        if span >= 0:
            print(f"  Answer span: token #{span} ({span / len(ct) * 100:.1f}% into passage)")
        else:
            print("  Answer span: not an exact token sequence (subword tokenization effect)")

    def key_findings(self, data) -> List[str]:
        return [
            "SQuAD is extractive — the answer is a literal span from the passage, "
            "so the n-gram drafter acts as exact span retrieval rather than guessing.",
            "Once the drafter finds a match its tokens are usually correct ('full_reject' is rare).",
            "'no_draft' steps come from subword context-dependence: a word tokenizes "
            "differently inside a sentence than standalone.",
            "Main mismatch cause is boundary overshoot — the drafter doesn't know where the "
            "answer ends and keeps going with passage context.",
        ]
