import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer
from specdecode.datasets import get_dataset

EXPERIMENTS = ["squad", "samsum", "xsum"]
COLORS = {"squad": "#3b82f6", "samsum": "#f59e0b", "xsum": "#ef4444"}
LABELS = {"squad": "SQuAD (Extractive)", "samsum": "SAMSum (Semi-extractive)", "xsum": "XSum (Abstractive)"}
MARKERS = {"squad": "o", "samsum": "s", "xsum": "^"}


def load_results(dataset: str) -> dict:
    path = os.path.join("experiments", dataset, "artifacts", "results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results found for '{dataset}'. Run: python run.py --dataset {dataset}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_lexical_overlap(corpus_tokens: list, target_tokens: list) -> float:
    """Fraction of target tokens that appear anywhere in the corpus."""
    corpus_set = set(corpus_tokens)
    matched = sum(1 for t in target_tokens if t in corpus_set)
    return matched / len(target_tokens) if target_tokens else 0.0


def main():
    all_results = {}
    for ds in EXPERIMENTS:
        all_results[ds] = load_results(ds)

    # Use the tokenizer from the first experiment (all share the same one)
    tokenizer_name = all_results[EXPERIMENTS[0]]["tokenizer"]
    print(f"Loading tokenizer ({tokenizer_name}) for overlap computation...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    overlaps = {}
    for ds in EXPERIMENTS:
        r = all_results[ds]
        corpus_text, target_text = get_dataset(ds, index=r["sample_index"])
        corpus_tokens = tokenizer.encode(corpus_text)
        target_tokens = tokenizer.encode(target_text)
        overlaps[ds] = compute_lexical_overlap(corpus_tokens, target_tokens)

    # ------------------------------------------------------------------ #
    # Print RCA summary table
    # ------------------------------------------------------------------ #
    print()
    print("=" * 75)
    print("ROOT CAUSE ANALYSIS — SUMMARY TABLE")
    print("=" * 75)
    header = f"{'Dataset':<20} {'Overlap':>9} {'Max Speedup':>12} {'Best K':>7} {'Max Avg Accept':>15} {'Accept Rate @K=3':>18}"
    print(header)
    print("-" * 75)
    for ds in EXPERIMENTS:
        sweep = all_results[ds]["sweep"]
        max_speedup = max(s["speedup"] for s in sweep)
        best_k = max(sweep, key=lambda s: s["speedup"])["k"]
        max_avg_acc = max(s["avg_accepted"] for s in sweep)
        k3_rate = next((s["acceptance_rate"] for s in sweep if s["k"] == 3), None)
        k3_str = f"{k3_rate:.1%}" if k3_rate is not None else "N/A"
        print(f"{ds:<20} {overlaps[ds]:>8.1%} {max_speedup:>11.2f}x {best_k:>7} {max_avg_acc:>14.2f} {k3_str:>18}")
    print("=" * 75)

    # ------------------------------------------------------------------ #
    # 4-panel combined chart
    # ------------------------------------------------------------------ #
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    pct_fmt = mticker.FuncFormatter(lambda y, _: f"{y:.0%}")

    # Panel 1 — Speedup vs K
    ax = axes[0][0]
    for ds in EXPERIMENTS:
        sweep = all_results[ds]["sweep"]
        ks   = [s["k"] for s in sweep]
        vals = [s["speedup"] for s in sweep]
        ax.plot(ks, vals, marker=MARKERS[ds], color=COLORS[ds], label=LABELS[ds], linewidth=2.5, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline (1.0x)")
    ax.set_title("Speedup Ratio vs Draft Size K", fontsize=13, fontweight="bold")
    ax.set_xlabel("Draft Size (K)")
    ax.set_ylabel("Speedup (x)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 2 — Avg accepted tokens vs K
    ax = axes[0][1]
    for ds in EXPERIMENTS:
        sweep = all_results[ds]["sweep"]
        ks   = [s["k"] for s in sweep]
        vals = [s["avg_accepted"] for s in sweep]
        ax.plot(ks, vals, marker=MARKERS[ds], color=COLORS[ds], label=LABELS[ds], linewidth=2.5, markersize=8)
    ax.set_title("Avg Accepted Tokens per Step vs K", fontsize=13, fontweight="bold")
    ax.set_xlabel("Draft Size (K)")
    ax.set_ylabel("Avg Accepted Tokens")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 3 — Acceptance rate vs K
    ax = axes[1][0]
    for ds in EXPERIMENTS:
        sweep = all_results[ds]["sweep"]
        ks   = [s["k"] for s in sweep]
        vals = [s["acceptance_rate"] for s in sweep]
        ax.plot(ks, vals, marker=MARKERS[ds], color=COLORS[ds], label=LABELS[ds], linewidth=2.5, markersize=8)
    ax.set_title("Draft Token Acceptance Rate vs K", fontsize=13, fontweight="bold")
    ax.set_xlabel("Draft Size (K)")
    ax.set_ylabel("Acceptance Rate")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 4 — Lexical overlap bar
    ax = axes[1][1]
    bars = ax.bar(
        [LABELS[ds] for ds in EXPERIMENTS],
        [overlaps[ds] for ds in EXPERIMENTS],
        color=[COLORS[ds] for ds in EXPERIMENTS],
        alpha=0.85, edgecolor="white", width=0.5,
    )
    for bar, ds in zip(bars, EXPERIMENTS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{overlaps[ds]:.1%}",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )
    ax.set_title("Lexical Overlap: Corpus ↔ Target", fontsize=13, fontweight="bold")
    ax.set_ylabel("% of target tokens found in corpus")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.set_ylim(0, 1.1)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.tick_params(axis="x", labelsize=9)

    plt.suptitle(
        "Speculative Decoding — Cross-Dataset Root Cause Analysis",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join("artifacts", "rca_combined.png")
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\nCombined RCA chart saved → {out_path}")


if __name__ == "__main__":
    main()
