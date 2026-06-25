"""
Cross-dataset summary — post-processes the per-dataset benchmark results that
``scripts/run_benchmark.py`` writes (experiments/<ds>/artifacts/results.json) into
one comparison table + a 4-panel chart. No simulations are re-run here.
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer

from specdecode.datasets import get_dataset
from .common import lexical_overlap

DEFAULT_DATASETS = ["squad", "samsum", "xsum"]
COLORS = {"squad": "#3b82f6", "samsum": "#f59e0b", "xsum": "#ef4444", "cnn_dailymail": "#10b981"}
LABELS = {"squad": "SQuAD (Extractive)", "samsum": "SAMSum (Semi-extractive)",
          "xsum": "XSum (Abstractive)", "cnn_dailymail": "CNN/DailyMail"}
MARKERS = {"squad": "o", "samsum": "s", "xsum": "^", "cnn_dailymail": "D"}


def _load_results(output_root: str, dataset: str) -> dict:
    path = os.path.join(output_root, "experiments", dataset, "artifacts", "results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No results for '{dataset}'. Run: python scripts/run_benchmark.py --dataset {dataset}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cross_dataset_summary(output_root: str = ".", datasets=None) -> None:
    datasets = datasets or DEFAULT_DATASETS
    results = {ds: _load_results(output_root, ds) for ds in datasets}

    tokenizer_name = results[datasets[0]]["tokenizer"]
    print(f"Loading tokenizer ({tokenizer_name}) for overlap computation...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    overlaps = {}
    for ds in datasets:
        corpus_text, target_text = get_dataset(ds, index=results[ds]["sample_index"])
        overlaps[ds] = lexical_overlap(tokenizer.encode(corpus_text), tokenizer.encode(target_text))

    print("\n" + "=" * 75)
    print("ROOT CAUSE ANALYSIS — CROSS-DATASET SUMMARY")
    print("=" * 75)
    print(f"{'Dataset':<22}{'Overlap':>9}{'Max Speedup':>13}{'Best K':>8}"
          f"{'Max Avg Accept':>16}{'Accept@K=3':>13}")
    print("-" * 75)
    for ds in datasets:
        sweep = results[ds]["sweep"]
        max_speedup = max(s["speedup"] for s in sweep)
        best_k = max(sweep, key=lambda s: s["speedup"])["k"]
        max_avg = max(s["avg_accepted"] for s in sweep)
        k3 = next((s["acceptance_rate"] for s in sweep if s["k"] == 3), None)
        k3_str = f"{k3:.1%}" if k3 is not None else "N/A"
        print(f"{ds:<22}{overlaps[ds]:>8.1%}{max_speedup:>12.2f}x{best_k:>8}{max_avg:>16.2f}{k3_str:>13}")
    print("=" * 75)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")
    pct_fmt = mticker.FuncFormatter(lambda y, _: f"{y:.0%}")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    def line_panel(ax, field, title, ylabel, as_pct=False):
        for ds in datasets:
            sweep = results[ds]["sweep"]
            ax.plot([s["k"] for s in sweep], [s[field] for s in sweep],
                    marker=MARKERS[ds], color=COLORS[ds], label=LABELS[ds],
                    linewidth=2.5, markersize=8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Draft Size (K)"); ax.set_ylabel(ylabel)
        if as_pct:
            ax.yaxis.set_major_formatter(pct_fmt)
        ax.legend(fontsize=9); ax.grid(True, linestyle=":", alpha=0.6)

    line_panel(axes[0][0], "speedup", "Speedup Ratio vs Draft Size K", "Speedup (x)")
    axes[0][0].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    line_panel(axes[0][1], "avg_accepted", "Avg Accepted Tokens per Step vs K", "Avg Accepted Tokens")
    line_panel(axes[1][0], "acceptance_rate", "Draft Token Acceptance Rate vs K", "Acceptance Rate", as_pct=True)

    ax = axes[1][1]
    bars = ax.bar([LABELS[ds] for ds in datasets], [overlaps[ds] for ds in datasets],
                  color=[COLORS[ds] for ds in datasets], alpha=0.85, edgecolor="white", width=0.5)
    for bar, ds in zip(bars, datasets):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{overlaps[ds]:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_title("Lexical Overlap: Corpus ↔ Target", fontsize=13, fontweight="bold")
    ax.set_ylabel("% of target tokens found in corpus")
    ax.yaxis.set_major_formatter(pct_fmt); ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y"); ax.tick_params(axis="x", labelsize=9)

    plt.suptitle("Speculative Decoding — Cross-Dataset Root Cause Analysis",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_dir = os.path.join(output_root, "experiments", "_summary", "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rca_combined.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nCombined RCA chart saved → {out_path}")
