"""CNN/DailyMail analyzer — multi-sentence highlights (mostly abstractive, entity-heavy)."""

from typing import List

from datasets import load_dataset

from .engine import DatasetAnalyzer


class CnnDailymailAnalyzer(DatasetAnalyzer):
    name = "cnn_dailymail"
    title = "CNN/DailyMail (Highlights)"
    color = "#10b981"
    marker = "D"
    fixed_n = 3
    max_draft = 6
    alignment_label = "Lexical Overlap"

    def load_raw(self):
        return load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")

    def extract(self, sample):
        return sample["article"], sample["highlights"]

    def key_findings(self, data) -> List[str]:
        return [
            "CNN/DailyMail highlights reuse many entities and phrases from the article but rewrite "
            "structure, so acceptance lands between extractive SQuAD and abstractive XSum.",
        ]
