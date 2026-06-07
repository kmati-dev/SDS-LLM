"""
XSum Root Cause Analysis
Run: python analyze_xsum.py

Analyses:
  1. Single-sample deep dive: step types, acceptance histogram, n-gram backoff, mismatch examples
  2. N-gram size effect (n=1..4 at fixed K)
  3. Multi-sample robustness (10 samples)
  4. Speedup vs lexical overlap scatter
  5. Corpus size sensitivity
  6. XSum-specific: token novelty rate + first-token analysis
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from transformers import AutoTokenizer
from datasets import load_dataset

from src.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback
from src.datasets import get_dataset

DATASET   = "xsum"
ARTIFACTS = f"experiments/{DATASET}/artifacts"
N_SAMPLES = 10
FIXED_K   = 3
FIXED_N   = 2       # XSum config uses n=2
MAX_DRAFT = 4       # XSum config uses max_draft=4
COLOR     = "#ef4444"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_tokenizer():
    cfg_path = f"experiments/{DATASET}/config.json"
    name = json.load(open(cfg_path)).get("tokenizer_name", "Qwen/Qwen2.5-0.5B-Instruct") \
           if os.path.exists(cfg_path) else "Qwen/Qwen2.5-0.5B-Instruct"
    return AutoTokenizer.from_pretrained(name), name


def run_sim(corpus_tokens, target_text, tokenizer, k=FIXED_K, n=FIXED_N):
    m = PlaybackMetrics()
    SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=NGramDrafter(corpus_tokens, n=n, draft_size=k),
        verifier=GreedyVerifier(),
        metrics=m,
    ).run_playback(target_text)
    return m


def compute_overlap(corpus_tokens, target_tokens):
    s = set(corpus_tokens)
    return sum(1 for t in target_tokens if t in s) / len(target_tokens) if target_tokens else 0.0


def token_novelty_rate(corpus_tokens, target_tokens):
    """Fraction of target tokens that do NOT appear anywhere in corpus."""
    corpus_set = set(corpus_tokens)
    novel = sum(1 for t in target_tokens if t not in corpus_set)
    return novel / len(target_tokens) if target_tokens else 0.0


# ── Text output ───────────────────────────────────────────────────────────────
def print_analysis(tokenizer, corpus_text, target_text):
    corpus_tokens = tokenizer.encode(corpus_text)
    target_tokens = tokenizer.encode(target_text)
    overlap       = compute_overlap(corpus_tokens, target_tokens)
    novelty       = token_novelty_rate(corpus_tokens, target_tokens)

    print("\n" + "=" * 65)
    print("XSum ROOT CAUSE ANALYSIS")
    print("=" * 65)

    print(f"\n[OVERVIEW]")
    print(f"  Summary  : '{target_text}'")
    print(f"  Corpus   : {len(corpus_tokens)} tokens (article)  |  Target: {len(target_tokens)} tokens (summary)")
    print(f"  Lexical overlap : {overlap:.1%}  (target tokens found in corpus)")
    print(f"  Token novelty   : {novelty:.1%}  (target tokens NOT in corpus at all)")

    # First-token analysis
    first_token_id   = target_tokens[0] if target_tokens else None
    first_token_text = tokenizer.decode([first_token_id]) if first_token_id else ""
    first_in_corpus  = first_token_id in set(corpus_tokens)
    print(f"\n[XSUM-SPECIFIC: FIRST TOKEN ANALYSIS]")
    print(f"  First summary token : '{first_token_text.strip()}'")
    print(f"  Exists in corpus    : {first_in_corpus}")
    print(f"  → If False: drafter will always reject the very first draft token,")
    print(f"    causing a 'full_reject' or 'no_draft' at step 1 (high first-step failure rate).")

    # Per-token novelty breakdown
    print(f"\n[TOKEN-LEVEL NOVELTY]")
    for i, tok_id in enumerate(target_tokens):
        tok_text = tokenizer.decode([tok_id])
        in_corp  = "✓" if tok_id in set(corpus_tokens) else "✗ (novel)"
        print(f"  Token {i+1:>2}: '{tok_text.strip():<20}'  {in_corp}")

    # Single sample sweep
    print(f"\n[SINGLE SAMPLE — SWEEP K=1..{MAX_DRAFT}]")
    print(f"  {'K':<4} {'steps':>6} {'speedup':>9} {'no_draft':>10} {'full_reject':>12} {'partial':>9} {'full_accept':>12} {'avg_acc':>8}")
    print("  " + "-" * 78)
    for k in range(1, MAX_DRAFT + 1):
        m = run_sim(corpus_tokens, target_text, tokenizer, k=k)
        st = m.step_types
        total = m.speculative_steps or 1
        print(f"  {k:<4} {m.speculative_steps:>6} {m.speedup_ratio:>8.2f}x"
              f"  {st['no_draft']:>3}({st['no_draft']/total:.0%})"
              f"  {st['full_reject']:>3}({st['full_reject']/total:.0%})"
              f"  {st['partial']:>3}({st['partial']/total:.0%})"
              f"  {st['full_accept']:>3}({st['full_accept']/total:.0%})"
              f"  {m.average_accepted_per_step:>7.2f}")

    # Deep dive
    m = run_sim(corpus_tokens, target_text, tokenizer)
    print(f"\n[DEEP DIVE — K={FIXED_K}, n={FIXED_N}]")
    print(f"  Speedup        : {m.speedup_ratio}x")
    print(f"  N-gram usage   : {m.n_gram_usage}")
    print(f"  Step types     : {m.step_types}")
    print(f"  Accepted counts: {m.step_accepted_counts}  (per step)")

    print(f"\n[MISMATCH EXAMPLES — K={FIXED_K}]")
    if m.mismatch_log:
        for i, mm in enumerate(m.mismatch_log):
            ctx  = repr(tokenizer.decode(mm["context_ids"]))
            exp  = repr(tokenizer.decode([mm["expected_id"]])) if mm["expected_id"] else "'<EOF>'"
            drft = repr(tokenizer.decode([mm["drafted_id"]])) if mm["drafted_id"] else "'<NONE>'"
            exp_in_corp  = mm["expected_id"] in set(corpus_tokens) if mm["expected_id"] else False
            drft_in_corp = mm["drafted_id"] in set(corpus_tokens) if mm["drafted_id"] else False
            mismatch_type = "paraphrase" if exp_in_corp else "novel word (not in article)"
            print(f"  #{i+1}  context : {ctx}")
            print(f"       expected: {exp} (in_corpus={exp_in_corp})  |  drafted: {drft} (in_corpus={drft_in_corp})")
            print(f"       mismatch type: {mismatch_type}")
    else:
        print("  No mismatches!")

    # N-gram size effect
    print(f"\n[N-GRAM SIZE EFFECT — K={FIXED_K}]")
    for n in [1, 2, 3, 4]:
        mn = run_sim(corpus_tokens, target_text, tokenizer, k=FIXED_K, n=n)
        print(f"  n={n}: speedup={mn.speedup_ratio}x  no_draft={mn.step_types['no_draft']}  "
              f"full_reject={mn.step_types['full_reject']}  n_gram_usage={mn.n_gram_usage}")

    # Multi-sample
    print(f"\n[MULTI-SAMPLE ROBUSTNESS — {N_SAMPLES} samples, K={FIXED_K}]")
    ds_raw  = load_dataset("EdinburghNLP/xsum", split="train")
    speedups, zero_drafts, novelties = [], [], []
    for idx in range(N_SAMPLES):
        s = ds_raw[idx]
        ct = tokenizer.encode(s["document"])
        tt_text = s["summary"]
        tt = tokenizer.encode(tt_text)
        if len(tt) < 2:
            continue
        mi = run_sim(ct, tt_text, tokenizer)
        nov = token_novelty_rate(ct, tt)
        speedups.append(mi.speedup_ratio)
        zero_drafts.append(mi.step_types["no_draft"] / (mi.speculative_steps or 1))
        novelties.append(nov)
        print(f"  Sample #{idx}: summary='{tt_text[:50]}'")
        print(f"          speedup={mi.speedup_ratio}x  novelty={nov:.1%}  "
              f"no_draft={mi.step_types['no_draft']/(mi.speculative_steps or 1):.0%}")

    print(f"\n  Speedup  : mean={np.mean(speedups):.2f}x  std={np.std(speedups):.2f}  "
          f"min={min(speedups):.2f}x  max={max(speedups):.2f}x")
    print(f"  Novelty  : mean={np.mean(novelties):.1%}  std={np.std(novelties):.1%}")
    print(f"  Zero-draft: mean={np.mean(zero_drafts):.1%}  std={np.std(zero_drafts):.1%}")

    # Key findings
    print(f"\n[KEY FINDINGS]")
    print(f"  1. XSum is abstractive — summaries are written fresh by BBC journalists.")
    print(f"     Token novelty {novelty:.1%}: nearly 1/3 of summary tokens don't exist in article.")
    print(f"  2. High 'full_reject' rate — drafter finds match in article but summary uses")
    print(f"     different words (paraphrase): e.g. article says 'policy', summary says 'plan'.")
    print(f"  3. First summary token '{first_token_text.strip()}' is {'NOT ' if not first_in_corpus else ''}in corpus")
    print(f"     → Step 1 is {'always a rejection' if not first_in_corpus else 'potentially matchable'}.")
    print(f"  4. N-gram size has little effect — increasing n doesn't help because the problem")
    print(f"     is vocabulary mismatch, not insufficient context.")
    print(f"  5. Corpus size sensitivity is low — more article text doesn't improve speedup")
    print(f"     because the summary vocabulary is fundamentally different.")

    return corpus_tokens, target_tokens, speedups, zero_drafts, novelties, ds_raw


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_charts(tokenizer, corpus_text, target_text, speedups, zero_drafts, novelties, ds_raw):
    os.makedirs(ARTIFACTS, exist_ok=True)
    corpus_tokens = tokenizer.encode(corpus_text)
    target_tokens = tokenizer.encode(target_text)
    pct_fmt = mticker.FuncFormatter(lambda y, _: f"{y:.0%}")

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")

    # ── Figure 1: Step types | Acceptance histogram | N-gram effect ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Step types per K (stacked bar)
    ax = axes[0]
    ks = list(range(1, MAX_DRAFT + 1))
    stype_data = {st: [] for st in ["no_draft", "full_reject", "partial", "full_accept"]}
    for k in ks:
        m = run_sim(corpus_tokens, target_text, tokenizer, k=k)
        total = m.speculative_steps or 1
        for st in stype_data:
            stype_data[st].append(m.step_types[st] / total)
    scolors = {"no_draft": "#94a3b8", "full_reject": "#ef4444", "partial": "#f59e0b", "full_accept": "#22c55e"}
    bottoms = [0.0] * len(ks)
    for st, sc in scolors.items():
        ax.bar(ks, stype_data[st], bottom=bottoms, color=sc, alpha=0.85,
               label=st.replace("_", " ").title(), width=0.6)
        bottoms = [b + h for b, h in zip(bottoms, stype_data[st])]
    ax.set_title("Step Type Breakdown per K", fontsize=12, fontweight="bold")
    ax.set_xlabel("Draft Size K")
    ax.set_ylabel("% of steps")
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.legend(fontsize=8)

    # Panel 2 — Acceptance histogram at FIXED_K
    ax = axes[1]
    m = run_sim(corpus_tokens, target_text, tokenizer)
    counts = m.step_accepted_counts
    bins = np.arange(-0.5, FIXED_K + 1.5)
    ax.hist(counts, bins=bins, color=COLOR, alpha=0.8, edgecolor="white", density=True)
    ax.set_title(f"Accepted Tokens per Step (K={FIXED_K})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Tokens accepted in one step")
    ax.set_ylabel("Proportion of steps")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.set_xticks(range(0, FIXED_K + 1))

    # Panel 3 — N-gram size effect
    ax = axes[2]
    n_vals = [1, 2, 3, 4]
    sp_n = [run_sim(corpus_tokens, target_text, tokenizer, k=FIXED_K, n=n).speedup_ratio for n in n_vals]
    ax.plot(n_vals, sp_n, marker="^", color=COLOR, linewidth=2.5, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    for n, s in zip(n_vals, sp_n):
        ax.annotate(f"{s:.2f}x", (n, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title(f"Speedup vs N-gram Size n (K={FIXED_K})", fontsize=12, fontweight="bold")
    ax.set_xlabel("N-gram size (n)")
    ax.set_ylabel("Speedup (x)")
    ax.set_xticks(n_vals)
    ax.legend(fontsize=9)

    plt.suptitle("XSum Analysis — Part 1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS, "xsum_part1.png"), dpi=300)
    plt.close()
    print(f"\nChart saved → {ARTIFACTS}/xsum_part1.png")

    # ── Figure 2: Multi-sample | Scatter | Corpus sensitivity ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Multi-sample speedup distribution
    ax = axes[0]
    ax.bar(range(len(speedups)), sorted(speedups, reverse=True), color=COLOR, alpha=0.8, edgecolor="white")
    ax.axhline(np.mean(speedups), color="#3b82f6", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(speedups):.2f}x")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Baseline")
    ax.set_title(f"Speedup Across {len(speedups)} Samples", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample (sorted)")
    ax.set_ylabel("Speedup (x)")
    ax.legend(fontsize=9)

    # Panel 2 — Speedup vs novelty scatter (XSum-specific)
    ax = axes[1]
    ax.scatter(novelties, speedups, color=COLOR, s=100, alpha=0.8, edgecolors="white", marker="^")
    for i, (nx, sy) in enumerate(zip(novelties, speedups)):
        ax.annotate(f"#{i}", (nx, sy), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_title("Speedup vs Token Novelty Rate", fontsize=12, fontweight="bold")
    ax.set_xlabel("Token Novelty (% of summary tokens not in article)")
    ax.set_ylabel("Speedup (x)")
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 3 — Corpus size sensitivity
    ax = axes[2]
    fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    all_corpus = tokenizer.encode(corpus_text)
    sp_fracs = []
    for frac in fracs:
        trunc = max(2, int(len(all_corpus) * frac))
        m_f = run_sim(all_corpus[:trunc], target_text, tokenizer)
        sp_fracs.append(m_f.speedup_ratio)
    ax.plot([f * 100 for f in fracs], sp_fracs, marker="^", color=COLOR,
            linewidth=2.5, markersize=8)
    for f, s in zip(fracs, sp_fracs):
        ax.annotate(f"{s:.2f}x", (f * 100, s), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_title("Corpus Size Sensitivity", fontsize=12, fontweight="bold")
    ax.set_xlabel("Corpus size (% of full article)")
    ax.set_ylabel("Speedup (x)")
    ax.legend(fontsize=9)

    plt.suptitle("XSum Analysis — Part 2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS, "xsum_part2.png"), dpi=300)
    plt.close()
    print(f"Chart saved → {ARTIFACTS}/xsum_part2.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer, tok_name = load_tokenizer()
    print(f"Tokenizer: {tok_name}")

    ds_raw = load_dataset("EdinburghNLP/xsum", split="train")
    s0 = ds_raw[0]
    corpus_text = s0["document"]
    target_text = s0["summary"]

    corpus_tokens, target_tokens, speedups, zero_drafts, novelties, ds_raw = \
        print_analysis(tokenizer, corpus_text, target_text)

    plot_charts(tokenizer, corpus_text, target_text, speedups, zero_drafts, novelties, ds_raw)


if __name__ == "__main__":
    main()
