"""XSum analyzer — abstractive single-sentence summarization (high token novelty)."""

from typing import List

from datasets import load_dataset

from .common import token_novelty_rate
from .engine import DatasetAnalyzer


class XsumAnalyzer(DatasetAnalyzer):
    name = "xsum"
    title = "XSum (Abstractive)"
    color = "#ef4444"
    marker = "^"
    fixed_n = 2
    max_draft = 4
    alignment_label = "Token Novelty"

    def load_raw(self):
        return load_dataset("EdinburghNLP/xsum", split="train")

    def extract(self, sample):
        return sample["document"], sample["summary"]

    def alignment_metric(self, corpus_tokens, target_tokens) -> float:
        return token_novelty_rate(corpus_tokens, target_tokens)

    def deepdive(self, tokenizer, ct: List[int], tt: List[int]) -> None:
        corpus_set = set(ct)
        if not tt:
            return
        first_text = tokenizer.decode([tt[0]]).strip()
        print(f"  First summary token: '{first_text}'  |  in corpus: {tt[0] in corpus_set}")
        print(f"  → if absent, step 1 is always a rejection (full_reject / no_draft).")
        novel = sum(1 for t in tt if t not in corpus_set)
        print(f"  Token novelty: {novel}/{len(tt)} summary tokens are not in the article.")

    def _mismatch_note(self, mm, corpus_set) -> str:
        exp_id = mm.get("expected_id")
        if exp_id is None:
            return ""
        in_corpus = exp_id in corpus_set
        kind = "paraphrase" if in_corpus else "novel word (not in article)"
        return f"  |  type: {kind}"

    def key_findings(self, data) -> List[str]:
        import numpy as np
        nov = np.mean(data["alignments"]) if data["alignments"] else 0.0
        return [
            f"XSum is abstractive — summaries are written fresh; mean token novelty {nov:.1%}.",
            "High 'full_reject' rate: the drafter finds a match in the article but the summary "
            "uses different words (paraphrase).",
            "N-gram size barely helps — the bottleneck is vocabulary mismatch, not context length.",
            "Corpus size sensitivity is low — more article text doesn't fix a different summary vocabulary.",
        ]
