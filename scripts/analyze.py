#!/usr/bin/env python3
"""
Unified analysis CLI for the speculative-decoding simulator.

    python scripts/analyze.py --dataset squad           # full RCA for one dataset
    python scripts/analyze.py --dataset xsum --limit 50  # quick run on first 50 samples
    python scripts/analyze.py --summary                  # cross-dataset comparison chart
    python scripts/analyze.py --wiki --lang lo           # low-resource tokenizer study

Requires the package on the path: either `pip install -e .` or `PYTHONPATH=src`.
Outputs go under <repo>/experiments/ regardless of the current working directory.
"""

import argparse
import os

# This script lives in <repo>/scripts/; anchor experiments/ outputs to the repo root.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    from specdecode.analysis import ANALYZERS

    p = argparse.ArgumentParser(
        prog="analyze.py",
        description="Speculative-decoding RCA: per-dataset, cross-dataset summary, or wiki tokenizer study.",
    )
    p.add_argument("--dataset", choices=sorted(ANALYZERS), help="run the full RCA for one dataset")
    p.add_argument(
        "--summary", action="store_true", help="cross-dataset summary from run_benchmark results"
    )
    p.add_argument(
        "--wiki",
        action="store_true",
        help="low-resource tokenizer study; remaining args are forwarded (e.g. --lang lo)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap #samples for a quick --dataset run (default: full dataset)",
    )
    p.add_argument(
        "--output-root",
        default=PROJECT_ROOT,
        help="repo root under which experiments/ and artifacts/ are written",
    )
    return p


def main():
    args, extra = build_parser().parse_known_args()

    if args.wiki:
        from specdecode.analysis import wiki

        wiki.main(extra, output_root=args.output_root)
    elif args.summary:
        from specdecode.analysis import cross_dataset_summary

        cross_dataset_summary(args.output_root)
    elif args.dataset:
        from specdecode.analysis import ANALYZERS

        ANALYZERS[args.dataset](output_root=args.output_root).run(limit=args.limit)
    else:
        build_parser().error("choose one of --dataset / --summary / --wiki")


if __name__ == "__main__":
    main()
