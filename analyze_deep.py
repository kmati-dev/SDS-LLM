"""
Deep RCA Analysis — SQuAD vs XSum
Analyses:
  Part 1: Step-type breakdown, Acceptance histogram, N-gram size effect
  Part 2: Multi-sample robustness (10 samples), Speedup vs overlap scatter, Corpus size sensitivity
"""
import os
import json
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer

from specdecode.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback
from specdecode.datasets import get_dataset

# ── Constants ─────────────────────────────────────────────────────────────────
DATASETS  = ["squad", "xsum"]
COLORS    = {"squad": "#3b82f6", "xsum": "#ef4444"}
LABELS    = {"squad": "SQuAD (Extractive)", "xsum": "XSum (Abstractive)"}
MARKERS   = {"squad": "o", "xsum": "^"}
N_SAMPLES = 10
FIXED_K   = 3
FIXED_N   = 3
ARTIFACTS = "artifacts"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_tokenizer() -> AutoTokenizer:
    cfg = {}
    for ds in DATASETS:
        p = f"experiments/{ds}/config.json"
        if os.path.exists(p):
            cfg = json.load(open(p))
            break
    return AutoTokenizer.from_pretrained(cfg.get("tokenizer_name", "Qwen/Qwen2.5-0.5B-Instruct"))


def compute_overlap(corpus_tokens: List[int], target_tokens: List[int]) -> float:
    s = set(corpus_tokens)
    return sum(1 for t in target_tokens if t in s) / len(target_tokens) if target_tokens else 0.0


def run_sim(corpus_tokens, target_text, tokenizer, k=FIXED_K, n=FIXED_N) -> PlaybackMetrics:
    m = PlaybackMetrics()
    SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens, n=n, draft_size=k),
        verifier=GreedyVerifier(),
        metrics=m,
    ).run_playback(target_text)
    return m


# ── Analysis functions ────────────────────────────────────────────────────────
def compute_step_types(tokenizer, sample0: dict) -> dict:
    result = {}
    for ds in DATASETS:
        corpus_tokens = tokenizer.encode(sample0[ds][0])
        m = run_sim(corpus_tokens, sample0[ds][1], tokenizer)
        total = m.speculative_steps or 1
        result[ds] = {k: v / total for k, v in m.step_types.items()}
    return result


def compute_acceptance_hist(tokenizer, sample0: dict) -> dict:
    result = {}
    for ds in DATASETS:
        corpus_tokens = tokenizer.encode(sample0[ds][0])
        m = run_sim(corpus_tokens, sample0[ds][1], tokenizer)
        result[ds] = m.step_accepted_counts
    return result


def compute_ngram_effect(tokenizer, sample0: dict):
    n_vals = [1, 2, 3, 4]
    result = {}
    for ds in DATASETS:
        corpus_tokens = tokenizer.encode(sample0[ds][0])
        result[ds] = [
            run_sim(corpus_tokens, sample0[ds][1], tokenizer, k=FIXED_K, n=n).speedup_ratio
            for n in n_vals
        ]
    return result, n_vals


def compute_multi_sample(tokenizer):
    result = {ds: {"speedups": [], "zero_draft": []} for ds in DATASETS}
    for ds in DATASETS:
        for idx in range(N_SAMPLES):
            try:
                corpus_text, target_text = get_dataset(ds, index=idx)
                corpus_tokens = tokenizer.encode(corpus_text)
                if len(tokenizer.encode(target_text)) < 3:
                    continue
                m = run_sim(corpus_tokens, target_text, tokenizer)
                result[ds]["speedups"].append(m.speedup_ratio)
                result[ds]["zero_draft"].append(m.step_types["no_draft"] / (m.speculative_steps or 1))
            except Exception:
                pass
    return result


def compute_scatter(tokenizer):
    result = {ds: {"overlaps": [], "speedups": []} for ds in DATASETS}
    for ds in DATASETS:
        for idx in range(N_SAMPLES):
            try:
                corpus_text, target_text = get_dataset(ds, index=idx)
                corpus_tokens = tokenizer.encode(corpus_text)
                target_tokens = tokenizer.encode(target_text)
                if len(target_tokens) < 3:
                    continue
                result[ds]["overlaps"].append(compute_overlap(corpus_tokens, target_tokens))
                result[ds]["speedups"].append(run_sim(corpus_tokens, target_text, tokenizer).speedup_ratio)
            except Exception:
                pass
    return result


def compute_corpus_sensitivity(tokenizer):
    fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    result = {ds: {"fracs": fracs, "speedups": [], "zero_drafts": []} for ds in DATASETS}
    for ds in DATASETS:
        for frac in fracs:
            sp, zd = [], []
            for idx in range(5):
                try:
                    corpus_text, target_text = get_dataset(ds, index=idx)
                    all_tok = tokenizer.encode(corpus_text)
                    corp = all_tok[:max(2, int(len(all_tok) * frac))]
                    if len(tokenizer.encode(target_text)) < 3:
                        continue
                    m = run_sim(corp, target_text, tokenizer)
                    sp.append(m.speedup_ratio)
                    zd.append(m.step_types["no_draft"] / (m.speculative_steps or 1))
                except Exception:
                    pass
            result[ds]["speedups"].append(float(np.mean(sp)) if sp else 1.0)
            result[ds]["zero_drafts"].append(float(np.mean(zd)) if zd else 1.0)
    return result


def print_mismatch_examples(tokenizer, sample0: dict):
    print("\n" + "=" * 65)
    print(f"MISMATCH TOKEN EXAMPLES  (K={FIXED_K}, n={FIXED_N})")
    print("=" * 65)
    for ds in DATASETS:
        corpus_tokens = tokenizer.encode(sample0[ds][0])
        m = run_sim(corpus_tokens, sample0[ds][1], tokenizer)
        st = m.step_types
        total = m.speculative_steps or 1
        print(f"\n[{ds.upper()}]")
        print(f"  step_types : no_draft={st['no_draft']/total:.0%}  full_reject={st['full_reject']/total:.0%}  "
              f"partial={st['partial']/total:.0%}  full_accept={st['full_accept']/total:.0%}")
        print(f"  n_gram_use : {m.n_gram_usage}")
        if m.mismatch_log:
            for i, mm in enumerate(m.mismatch_log[:3]):
                ctx  = repr(tokenizer.decode(mm["context_ids"]))
                exp  = repr(tokenizer.decode([mm["expected_id"]])) if mm["expected_id"] else "'<EOF>'"
                drft = repr(tokenizer.decode([mm["drafted_id"]])) if mm["drafted_id"] else "'<NONE>'"
                print(f"  Mismatch #{i+1}: context={ctx}")
                print(f"             expected={exp}  drafted={drft}  accepted_before={mm['accepted_count']}  n_used={mm['n_used']}")
        else:
            print("  → No mismatches (all drafts fully accepted)")


# ── Plotting ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(ARTIFACTS, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading sample #0 for SQuAD and XSum...")
    sample0 = {ds: get_dataset(ds, index=0) for ds in DATASETS}

    print_mismatch_examples(tokenizer, sample0)

    print("\nRunning analyses...")
    step_types_data          = compute_step_types(tokenizer, sample0)
    acceptance_data          = compute_acceptance_hist(tokenizer, sample0)
    ngram_data, n_vals       = compute_ngram_effect(tokenizer, sample0)
    print(f"  multi-sample ({N_SAMPLES} samples)...")
    multi_data               = compute_multi_sample(tokenizer)
    scatter_data             = compute_scatter(tokenizer)
    print("  corpus size sensitivity...")
    sensitivity_data         = compute_corpus_sensitivity(tokenizer)

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")

    pct_fmt = mticker.FuncFormatter(lambda y, _: f"{y:.0%}")

    # ── Figure 1: Step types | Acceptance histogram | N-gram effect ──────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Step types grouped bar
    ax = axes[0]
    stypes  = ["no_draft", "full_reject", "partial", "full_accept"]
    scolors = ["#94a3b8",  "#ef4444",     "#f59e0b", "#22c55e"]
    x, w = np.arange(len(DATASETS)), 0.18
    for i, (stype, sc) in enumerate(zip(stypes, scolors)):
        vals = [step_types_data[ds].get(stype, 0) for ds in DATASETS]
        ax.bar(x + i * w, vals, w, color=sc, alpha=0.85,
               label=stype.replace("_", " ").title())
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([LABELS[ds].split()[0] for ds in DATASETS])
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.set_title(f"Step Type Breakdown (K={FIXED_K})", fontsize=12, fontweight="bold")
    ax.set_ylabel("% of steps")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 2 — Acceptance histogram
    ax = axes[1]
    bins = np.arange(-0.5, FIXED_K + 1.5)
    for ds in DATASETS:
        counts = acceptance_data[ds]
        if counts:
            ax.hist(counts, bins=bins, alpha=0.6, color=COLORS[ds],
                    label=LABELS[ds].split()[0], density=True, edgecolor="white")
    ax.set_title(f"Accepted Tokens per Step (K={FIXED_K})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tokens accepted in one step")
    ax.set_ylabel("Proportion of steps")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.set_xticks(range(0, FIXED_K + 1))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 3 — N-gram size effect
    ax = axes[2]
    for ds in DATASETS:
        ax.plot(n_vals, ngram_data[ds], marker=MARKERS[ds], color=COLORS[ds],
                label=LABELS[ds].split()[0], linewidth=2.5, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_title(f"Speedup vs N-gram Size n (K={FIXED_K})", fontsize=12, fontweight="bold")
    ax.set_xlabel("N-gram size (n)")
    ax.set_ylabel("Speedup (x)")
    ax.set_xticks(n_vals)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Deep RCA Part 1 — SQuAD vs XSum", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out1 = os.path.join(ARTIFACTS, "rca_deep_part1.png")
    plt.savefig(out1, dpi=300)
    print(f"\nFigure 1 saved → {out1}")
    plt.close()

    # ── Figure 2: Multi-sample | Scatter | Corpus sensitivity ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Multi-sample bar with error bars
    ax = axes[0]
    means = [np.mean(multi_data[ds]["speedups"]) if multi_data[ds]["speedups"] else 1.0 for ds in DATASETS]
    stds  = [np.std(multi_data[ds]["speedups"])  if multi_data[ds]["speedups"] else 0.0 for ds in DATASETS]
    bars = ax.bar([LABELS[ds].split()[0] for ds in DATASETS], means,
                  color=[COLORS[ds] for ds in DATASETS], alpha=0.85, width=0.5,
                  yerr=stds, capsize=8, error_kw={"linewidth": 2})
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.05,
                f"{m:.2f}x", ha="center", va="bottom", fontweight="bold")
    ax.set_title(f"Speedup — {N_SAMPLES} Samples (mean ± std)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Speedup (x)")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Panel 2 — Speedup vs lexical overlap scatter
    ax = axes[1]
    for ds in DATASETS:
        ax.scatter(scatter_data[ds]["overlaps"], scatter_data[ds]["speedups"],
                   color=COLORS[ds], label=LABELS[ds].split()[0],
                   marker=MARKERS[ds], s=100, alpha=0.8, edgecolors="white")
    ax.set_title("Speedup vs Lexical Overlap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Lexical Overlap (corpus ↔ target)")
    ax.set_ylabel("Speedup (x)")
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 3 — Corpus size sensitivity
    ax = axes[2]
    fracs = sensitivity_data[DATASETS[0]]["fracs"]
    for ds in DATASETS:
        ax.plot([f * 100 for f in fracs], sensitivity_data[ds]["speedups"],
                marker=MARKERS[ds], color=COLORS[ds],
                label=LABELS[ds].split()[0], linewidth=2.5, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_title("Corpus Size Sensitivity", fontsize=12, fontweight="bold")
    ax.set_xlabel("Corpus size (% of full)")
    ax.set_ylabel("Speedup (x)")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Deep RCA Part 2 — SQuAD vs XSum", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out2 = os.path.join(ARTIFACTS, "rca_deep_part2.png")
    plt.savefig(out2, dpi=300)
    print(f"Figure 2 saved → {out2}")
    plt.close()


if __name__ == "__main__":
    main()
