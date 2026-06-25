"""
XSum Root Cause Analysis
Run: python analyze_xsum.py

Analyses:
  1. Single-sample deep dive: step types, acceptance histogram, n-gram backoff, mismatch examples
  2. N-gram size effect (n=1..4 at fixed K)  — averaged over full dataset
  3. Multi-sample robustness (full dataset)
  4. Speedup vs token novelty scatter         — full dataset
  5. Corpus size sensitivity                  — averaged over full dataset
  6. XSum-specific: token novelty rate + first-token analysis
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

from specdecode.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback
from specdecode.datasets import get_dataset

DATASET          = "xsum"
ARTIFACTS        = f"experiments/{DATASET}/artifacts"
CHART_CHECKPOINT = f"experiments/{DATASET}/full_analysis.json"
FIXED_K          = 3
FIXED_N          = 2
MAX_DRAFT        = 4
COLOR            = "#ef4444"
SCATTER_MAX      = 500
CKPT_INTERVAL    = 1000
FRACS            = [0.1, 0.25, 0.5, 0.75, 1.0]
N_VALS           = [1, 2, 3, 4]


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
    corpus_set = set(corpus_tokens)
    novel = sum(1 for t in target_tokens if t not in corpus_set)
    return novel / len(target_tokens) if target_tokens else 0.0


def _save_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


# ── Full-dataset computation (single pass, checkpointed) ─────────────────────
def compute_all_data(tokenizer, ds_raw):
    """
    One pass over the full dataset. Collects everything needed for both
    text output and all chart panels. Saves/resumes from CHART_CHECKPOINT.

    Returns a dict with:
      speedups, zero_drafts, novelties       — per-sample lists
      k_stype_means                          — {k: {step_type: mean_fraction}}
      n_speedup_means                        — {n: mean_speedup}
      accepted_hist                          — {count: total_occurrences}
      frac_speedup_means                     — {frac_str: mean_speedup}
      valid_count                            — int
    """
    n_total = len(ds_raw)

    # Running accumulators (string keys for JSON compatibility)
    speedups, zero_drafts, novelties = [], [], []
    k_stype_sums = {str(k): {"no_draft": 0.0, "full_reject": 0.0,
                              "partial": 0.0, "full_accept": 0.0}
                    for k in range(1, MAX_DRAFT + 1)}
    n_speedup_sums  = {str(n): 0.0 for n in N_VALS}
    accepted_hist   = {}                           # {count_str: total_occurrences}
    frac_speedup_sums = {str(f): 0.0 for f in FRACS}
    valid_count = 0
    start_idx   = 0

    if os.path.exists(CHART_CHECKPOINT):
        with open(CHART_CHECKPOINT) as f:
            ckpt = json.load(f)
        speedups          = ckpt.get("speedups", [])
        zero_drafts       = ckpt.get("zero_drafts", [])
        novelties         = ckpt.get("novelties", [])
        k_stype_sums      = ckpt.get("k_stype_sums", k_stype_sums)
        n_speedup_sums    = ckpt.get("n_speedup_sums", n_speedup_sums)
        accepted_hist     = ckpt.get("accepted_hist", {})
        frac_speedup_sums = ckpt.get("frac_speedup_sums", frac_speedup_sums)
        valid_count       = ckpt.get("valid_count", 0)
        start_idx         = ckpt.get("next_idx", 0)
        print(f"\n[FULL-DATASET PASS — resuming from idx={start_idx}/{n_total}]")
    else:
        print(f"\n[FULL-DATASET PASS — {n_total} samples]")

    for idx in tqdm(range(start_idx, n_total), desc="Computing", unit="sample",
                    initial=start_idx, total=n_total):
        s       = ds_raw[idx]
        ct      = tokenizer.encode(s["document"])
        tt_text = s["summary"]
        tt      = tokenizer.encode(tt_text)
        if len(tt) < 2:
            continue

        valid_count += 1

        # ── Basic multi-sample stats (FIXED_K, FIXED_N) ──
        mi  = run_sim(ct, tt_text, tokenizer)
        nov = token_novelty_rate(ct, tt)
        speedups.append(mi.speedup_ratio)
        zero_drafts.append(mi.step_types["no_draft"] / (mi.speculative_steps or 1))
        novelties.append(nov)

        # ── Acceptance histogram (pooled across all steps) ──
        for cnt in mi.step_accepted_counts:
            key = str(cnt)
            accepted_hist[key] = accepted_hist.get(key, 0) + 1

        # ── Step types per K ──
        for k in range(1, MAX_DRAFT + 1):
            mk    = run_sim(ct, tt_text, tokenizer, k=k)
            total = mk.speculative_steps or 1
            for st in ["no_draft", "full_reject", "partial", "full_accept"]:
                k_stype_sums[str(k)][st] += mk.step_types[st] / total

        # ── N-gram size effect ──
        for n in N_VALS:
            mn = run_sim(ct, tt_text, tokenizer, k=FIXED_K, n=n)
            n_speedup_sums[str(n)] += mn.speedup_ratio

        # ── Corpus size sensitivity ──
        for frac in FRACS:
            trunc = max(2, int(len(ct) * frac))
            mf    = run_sim(ct[:trunc], tt_text, tokenizer)
            frac_speedup_sums[str(frac)] += mf.speedup_ratio

        if (idx + 1) % CKPT_INTERVAL == 0:
            _save_checkpoint(CHART_CHECKPOINT, {
                "speedups": speedups, "zero_drafts": zero_drafts, "novelties": novelties,
                "k_stype_sums": k_stype_sums, "n_speedup_sums": n_speedup_sums,
                "accepted_hist": accepted_hist, "frac_speedup_sums": frac_speedup_sums,
                "valid_count": valid_count, "next_idx": idx + 1,
            })

    _save_checkpoint(CHART_CHECKPOINT, {
        "speedups": speedups, "zero_drafts": zero_drafts, "novelties": novelties,
        "k_stype_sums": k_stype_sums, "n_speedup_sums": n_speedup_sums,
        "accepted_hist": accepted_hist, "frac_speedup_sums": frac_speedup_sums,
        "valid_count": valid_count, "next_idx": n_total,
    })
    print(f"  Results saved → {CHART_CHECKPOINT}")

    n = valid_count or 1
    return {
        "speedups":          speedups,
        "zero_drafts":       zero_drafts,
        "novelties":         novelties,
        "k_stype_means":     {k: {st: v / n for st, v in stypes.items()}
                              for k, stypes in k_stype_sums.items()},
        "n_speedup_means":   {nk: v / n for nk, v in n_speedup_sums.items()},
        "accepted_hist":     accepted_hist,
        "frac_speedup_means":{fk: v / n for fk, v in frac_speedup_sums.items()},
        "valid_count":       valid_count,
    }


# ── Single-sample deep dive (text output only) ────────────────────────────────
def print_single_sample_analysis(tokenizer, corpus_text, target_text):
    corpus_tokens = tokenizer.encode(corpus_text)
    target_tokens = tokenizer.encode(target_text)
    overlap       = compute_overlap(corpus_tokens, target_tokens)
    novelty       = token_novelty_rate(corpus_tokens, target_tokens)

    print("\n" + "=" * 65)
    print("XSum ROOT CAUSE ANALYSIS  (deep dive — sample #0)")
    print("=" * 65)

    print(f"\n[OVERVIEW]")
    print(f"  Summary  : '{target_text}'")
    print(f"  Corpus   : {len(corpus_tokens)} tokens (article)  |  Target: {len(target_tokens)} tokens (summary)")
    print(f"  Lexical overlap : {overlap:.1%}  (target tokens found in corpus)")
    print(f"  Token novelty   : {novelty:.1%}  (target tokens NOT in corpus at all)")

    first_token_id   = target_tokens[0] if target_tokens else None
    first_token_text = tokenizer.decode([first_token_id]) if first_token_id else ""
    first_in_corpus  = first_token_id in set(corpus_tokens)
    print(f"\n[XSUM-SPECIFIC: FIRST TOKEN ANALYSIS]")
    print(f"  First summary token : '{first_token_text.strip()}'")
    print(f"  Exists in corpus    : {first_in_corpus}")
    print(f"  → If False: drafter will always reject the very first draft token,")
    print(f"    causing a 'full_reject' or 'no_draft' at step 1.")

    print(f"\n[TOKEN-LEVEL NOVELTY]")
    for i, tok_id in enumerate(target_tokens):
        tok_text = tokenizer.decode([tok_id])
        in_corp  = "✓" if tok_id in set(corpus_tokens) else "✗ (novel)"
        print(f"  Token {i+1:>2}: '{tok_text.strip():<20}'  {in_corp}")

    print(f"\n[SINGLE SAMPLE — SWEEP K=1..{MAX_DRAFT}]")
    print(f"  {'K':<4} {'steps':>6} {'speedup':>9} {'no_draft':>10} {'full_reject':>12} {'partial':>9} {'full_accept':>12} {'avg_acc':>8}")
    print("  " + "-" * 78)
    for k in range(1, MAX_DRAFT + 1):
        m     = run_sim(corpus_tokens, target_text, tokenizer, k=k)
        st    = m.step_types
        total = m.speculative_steps or 1
        print(f"  {k:<4} {m.speculative_steps:>6} {m.speedup_ratio:>8.2f}x"
              f"  {st['no_draft']:>3}({st['no_draft']/total:.0%})"
              f"  {st['full_reject']:>3}({st['full_reject']/total:.0%})"
              f"  {st['partial']:>3}({st['partial']/total:.0%})"
              f"  {st['full_accept']:>3}({st['full_accept']/total:.0%})"
              f"  {m.average_accepted_per_step:>7.2f}")

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
            mtype = "paraphrase" if exp_in_corp else "novel word (not in article)"
            print(f"  #{i+1}  context : {ctx}")
            print(f"       expected: {exp} (in_corpus={exp_in_corp})  |  drafted: {drft} (in_corpus={drft_in_corp})")
            print(f"       mismatch type: {mtype}")
    else:
        print("  No mismatches!")

    return novelty, first_token_text, first_in_corpus


# ── Multi-sample stats (text output) ─────────────────────────────────────────
def print_multi_sample_stats(data, novelty, first_token_text, first_in_corpus):
    speedups    = data["speedups"]
    novelties   = data["novelties"]
    zero_drafts = data["zero_drafts"]
    n           = data["valid_count"]

    print(f"\n[MULTI-SAMPLE ROBUSTNESS — {n:,} samples, K={FIXED_K}]")
    print(f"  Speedup  : mean={np.mean(speedups):.2f}x  std={np.std(speedups):.2f}  "
          f"min={min(speedups):.2f}x  max={max(speedups):.2f}x")
    print(f"  Novelty  : mean={np.mean(novelties):.1%}  std={np.std(novelties):.1%}")
    print(f"  Zero-draft: mean={np.mean(zero_drafts):.1%}  std={np.std(zero_drafts):.1%}")

    print(f"\n[N-GRAM SIZE EFFECT — K={FIXED_K}, averaged over {n:,} samples]")
    for nk, sp in sorted(data["n_speedup_means"].items(), key=lambda x: int(x[0])):
        print(f"  n={nk}: mean_speedup={sp:.3f}x")

    print(f"\n[KEY FINDINGS]")
    print(f"  1. XSum is abstractive — summaries are written fresh by BBC journalists.")
    print(f"     Mean token novelty {np.mean(novelties):.1%} across {n:,} samples.")
    print(f"  2. High 'full_reject' rate — drafter finds match in article but summary uses")
    print(f"     different words (paraphrase): e.g. article says 'policy', summary says 'plan'.")
    print(f"  3. First summary token '{first_token_text.strip()}' is {'NOT ' if not first_in_corpus else ''}in corpus")
    print(f"     → Step 1 is {'always a rejection' if not first_in_corpus else 'potentially matchable'}.")
    print(f"  4. N-gram size has little effect — increasing n doesn't help because the problem")
    print(f"     is vocabulary mismatch, not insufficient context.")
    print(f"  5. Corpus size sensitivity is low — more article text doesn't improve speedup")
    print(f"     because the summary vocabulary is fundamentally different.")


# ── Charts (all panels use full-dataset data) ─────────────────────────────────
def plot_charts(data):
    os.makedirs(ARTIFACTS, exist_ok=True)
    pct_fmt = mticker.FuncFormatter(lambda y, _: f"{y:.0%}")

    speedups    = data["speedups"]
    novelties   = data["novelties"]
    n_samples   = data["valid_count"]

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")

    # ── Figure 1: Step types | Acceptance histogram | N-gram effect ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Step types per K (stacked bar, full-dataset mean)
    ax = axes[0]
    ks = list(range(1, MAX_DRAFT + 1))
    stype_data = {st: [] for st in ["no_draft", "full_reject", "partial", "full_accept"]}
    for k in ks:
        means = data["k_stype_means"][str(k)]
        for st in stype_data:
            stype_data[st].append(means[st])
    scolors = {"no_draft": "#94a3b8", "full_reject": "#ef4444",
               "partial": "#f59e0b", "full_accept": "#22c55e"}
    bottoms = [0.0] * len(ks)
    for st, sc in scolors.items():
        ax.bar(ks, stype_data[st], bottom=bottoms, color=sc, alpha=0.85,
               label=st.replace("_", " ").title(), width=0.6)
        bottoms = [b + h for b, h in zip(bottoms, stype_data[st])]
    ax.set_title(f"Step Type Breakdown per K\n(mean over {n_samples:,} samples)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Draft Size K")
    ax.set_ylabel("Mean % of steps")
    ax.set_xticks(ks)
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.legend(fontsize=8)

    # Panel 2 — Acceptance histogram (pooled across full dataset)
    ax = axes[1]
    hist = data["accepted_hist"]
    total_steps = sum(hist.values())
    xs = list(range(0, FIXED_K + 1))
    ys = [hist.get(str(x), 0) / total_steps for x in xs]
    ax.bar(xs, ys, color=COLOR, alpha=0.8, edgecolor="white", width=0.6)
    ax.set_title(f"Accepted Tokens per Step (K={FIXED_K})\n(pooled over {n_samples:,} samples)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Tokens accepted in one step")
    ax.set_ylabel("Proportion of steps")
    ax.yaxis.set_major_formatter(pct_fmt)
    ax.set_xticks(xs)

    # Panel 3 — N-gram size effect (full-dataset mean speedup)
    ax = axes[2]
    sp_n = [data["n_speedup_means"][str(n)] for n in N_VALS]
    ax.plot(N_VALS, sp_n, marker="^", color=COLOR, linewidth=2.5, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    for n, s in zip(N_VALS, sp_n):
        ax.annotate(f"{s:.2f}x", (n, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_title(f"Mean Speedup vs N-gram Size n (K={FIXED_K})\n(mean over {n_samples:,} samples)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("N-gram size (n)")
    ax.set_ylabel("Mean Speedup (x)")
    ax.set_xticks(N_VALS)
    ax.legend(fontsize=9)

    plt.suptitle("XSum Analysis — Part 1 (full dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS, "xsum_part1.png"), dpi=300)
    plt.close()
    print(f"\nChart saved → {ARTIFACTS}/xsum_part1.png")

    # ── Figure 2: Speedup distribution | Scatter | Corpus sensitivity ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1 — Speedup distribution
    ax = axes[0]
    ax.bar(range(len(speedups)), sorted(speedups, reverse=True),
           color=COLOR, alpha=0.8, edgecolor="none", width=1.0)
    ax.axhline(np.mean(speedups), color="#3b82f6", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(speedups):.2f}x")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Baseline")
    ax.set_title(f"Speedup Across {n_samples:,} Samples", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sample (sorted by speedup)")
    ax.set_ylabel("Speedup (x)")
    ax.legend(fontsize=9)

    # Panel 2 — Speedup vs novelty scatter (subsampled for visibility)
    ax = axes[1]
    rng = np.random.default_rng(42)
    n_scatter   = min(SCATTER_MAX, len(novelties))
    scatter_idx = rng.choice(len(novelties), size=n_scatter, replace=False)
    sc_nov = [novelties[i] for i in scatter_idx]
    sc_sp  = [speedups[i]  for i in scatter_idx]
    ax.scatter(sc_nov, sc_sp, color=COLOR, s=30, alpha=0.5, edgecolors="none", marker="^")
    ax.set_title(f"Speedup vs Token Novelty Rate\n({n_scatter} sampled from {n_samples:,})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Token Novelty (% of summary tokens not in article)")
    ax.set_ylabel("Speedup (x)")
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Panel 3 — Corpus size sensitivity (full-dataset mean)
    ax = axes[2]
    sp_fracs = [data["frac_speedup_means"][str(f)] for f in FRACS]
    ax.plot([f * 100 for f in FRACS], sp_fracs, marker="^", color=COLOR,
            linewidth=2.5, markersize=8)
    for f, s in zip(FRACS, sp_fracs):
        ax.annotate(f"{s:.2f}x", (f * 100, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_title(f"Corpus Size Sensitivity\n(mean over {n_samples:,} samples)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Corpus size (% of full article)")
    ax.set_ylabel("Mean Speedup (x)")
    ax.legend(fontsize=9)

    plt.suptitle("XSum Analysis — Part 2 (full dataset)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS, "xsum_part2.png"), dpi=300)
    plt.close()
    print(f"Chart saved → {ARTIFACTS}/xsum_part2.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer, tok_name = load_tokenizer()
    print(f"Tokenizer: {tok_name}")

    ds_raw = load_dataset("EdinburghNLP/xsum", split="train")
    s0          = ds_raw[0]
    corpus_text = s0["document"]
    target_text = s0["summary"]

    # Full-dataset pass (resumes from checkpoint if exists)
    data = compute_all_data(tokenizer, ds_raw)

    # Text output
    novelty, first_token_text, first_in_corpus = \
        print_single_sample_analysis(tokenizer, corpus_text, target_text)
    print_multi_sample_stats(data, novelty, first_token_text, first_in_corpus)

    # Charts (all full-dataset)
    plot_charts(data)


if __name__ == "__main__":
    main()
