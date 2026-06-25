"""
Root-cause-analysis harness for greedy speculative decoding.

- ``DatasetAnalyzer`` + the per-dataset subclasses below run a full-dataset RCA
  for one dataset (use the ``ANALYZERS`` registry to look one up by name).
- ``cross_dataset_summary`` builds the cross-dataset comparison from the
  per-dataset benchmark results that ``scripts/run_benchmark.py`` writes.
- ``analysis.wiki`` is a separate low-resource tokenizer study (tensor depth/width
  drafting across tokenizers) — imported lazily by the CLI because it is heavier.
"""

from .engine import DatasetAnalyzer
from .squad import SquadAnalyzer
from .xsum import XsumAnalyzer
from .samsum import SamsumAnalyzer
from .cnn_dailymail import CnnDailymailAnalyzer
from .summary import cross_dataset_summary

ANALYZERS = {
    "squad": SquadAnalyzer,
    "xsum": XsumAnalyzer,
    "samsum": SamsumAnalyzer,
    "cnn_dailymail": CnnDailymailAnalyzer,
}

__all__ = [
    "DatasetAnalyzer",
    "ANALYZERS",
    "cross_dataset_summary",
    "SquadAnalyzer",
    "XsumAnalyzer",
    "SamsumAnalyzer",
    "CnnDailymailAnalyzer",
]
