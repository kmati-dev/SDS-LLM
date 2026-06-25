"""SAMSum analyzer — semi-extractive dialogue summarization."""

from typing import List

from datasets import load_dataset

from .engine import DatasetAnalyzer


class SamsumAnalyzer(DatasetAnalyzer):
    name = "samsum"
    title = "SAMSum (Semi-extractive)"
    color = "#f59e0b"
    marker = "s"
    fixed_n = 3
    max_draft = 6
    alignment_label = "Lexical Overlap"

    def load_raw(self):
        return load_dataset("knkarthick/samsum", split="train")

    def extract(self, sample):
        return sample["dialogue"], sample["summary"]

    def key_findings(self, data) -> List[str]:
        return [
            "SAMSum is semi-extractive — summaries reuse key phrases from the dialogue but also "
            "paraphrase and compress, so it sits between extractive SQuAD and abstractive XSum.",
        ]
