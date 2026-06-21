"""
Low-Resource Tokenizer RCA via Greedy Speculative Decoding
==========================================================
Run (Lao):   /opt/miniconda3/bin/python3.13 analyze_wiki.py --lang lo
Other langs: ... --lang my   (Burmese) | --lang ar | --lang ru | --lang uk
             (a teammate changes ONLY --lang — everything else is language-agnostic)

What it studies — how modern tokenizers (Qwen 3.5 vs Gemma 4) handle a low-resource
language, using an n-gram speculative drafter as the measurement instrument:

  A. Tokenizer fertility & fragmentation (static): tokens/char, tokens/byte,
     byte-fragment rate, single-character coverage.
  B. Speculative speedup, step-type breakdown, n-gram coverage.
  C. Corpus-size sensitivity (the assignment's 1M/10M/100M axis, capped to what exists).
  D. Depth-draft vs Width-draft under a fixed token budget (the feat/tensor axis).
  E. Effective generation rate = source characters produced per target-model step
     (the fair cross-tokenizer metric that neutralises fertility inflation).

Everything is printed to stdout, saved to experiments/wiki_<lang>/full_analysis.json,
and plotted to experiments/wiki_<lang>/artifacts/*.png.
"""

import os
import json
import argparse
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer

from src.datasets.wiki import load_articles
from src.simulator import (
    NGramIndex,
    IndexedTensorNGramDrafter,
    TensorGreedyVerifier,
    TensorSpeculativePlayback,
    PlaybackMetrics,
)

# ── Tokenizer registry (Qwen + Gemma per assignment) ──────────────────────────
TOKENIZERS = {
    "qwen":  "Qwen/Qwen3.5-4B",
    "gemma": "google/gemma-4-31B-it",
}
REPLACEMENT = "�"


# ── Helpers ───────────────────────────────────────────────────────────────────
def encode(tok, text: str) -> List[int]:
    """Clean content tokens (no BOS/EOS) — used for corpus and targets alike."""
    return tok.encode(text, add_special_tokens=False)


def lao_like_codepoints(lang: str) -> List[str]:
    """Representative single characters of the target script, for coverage testing."""
    blocks = {
        "lo": (0x0E80, 0x0EFF),   # Lao
        "my": (0x1000, 0x109F),   # Myanmar
        "ar": (0x0600, 0x06FF),   # Arabic
        "ru": (0x0400, 0x04FF),   # Cyrillic
        "uk": (0x0400, 0x04FF),   # Cyrillic
    }
    lo, hi = blocks.get(lang, (0x0E80, 0x0EFF))
    chars = []
    for cp in range(lo, hi + 1):
        c = chr(cp)
        if unicodedata.category(c)[0] in ("L", "M", "N"):  # letters/marks/numbers
            chars.append(c)
    return chars


def fertility_and_fragmentation(tok, sample_text: str, lang: str) -> Dict:
    """Static tokenizer metrics on a fixed text sample (no simulation)."""
    ids = encode(tok, sample_text)
    n_tok = len(ids)
    n_char = len(sample_text)
    n_byte = len(sample_text.encode("utf-8"))

    # Byte-fragment rate: a token whose standalone decode contains the U+FFFD
    # replacement char is an incomplete-UTF-8 (byte-fallback) fragment, not a
    # real sub-word. Computed on the (already small) sample's unique ids.
    frag = 0
    uniq = set(ids)
    id_decode = {i: tok.decode([i]) for i in uniq}
    for i in ids:
        if REPLACEMENT in id_decode[i]:
            frag += 1

    # Single-character coverage of the script.
    script_chars = lao_like_codepoints(lang)
    one_token = sum(1 for c in script_chars if len(encode(tok, c)) == 1)
    coverage = one_token / len(script_chars) if script_chars else 0.0

    # Token char-length distribution (how many chars a token spans).
    lengths = [len(id_decode[i]) for i in uniq]

    return {
        "tokens": n_tok,
        "chars": n_char,
        "bytes": n_byte,
        "tokens_per_char": n_tok / n_char if n_char else 0.0,
        "tokens_per_byte": n_tok / n_byte if n_byte else 0.0,
        "chars_per_token": n_char / n_tok if n_tok else 0.0,
        "byte_fragment_rate": frag / n_tok if n_tok else 0.0,
        "script_single_char_coverage": coverage,
        "script_chars_tested": len(script_chars),
        "median_token_chars": float(np.median(lengths)) if lengths else 0.0,
    }


def run_targets(
    tok,
    drafter,
    targets: List[Tuple[List[int], int]],
    collect_mismatch: bool = False,
) -> Dict:
    """
    Run the tensor speculative playback over a set of (token_ids, char_len) targets
    using a fixed drafter config. Returns aggregated metrics.
    """
    verifier = TensorGreedyVerifier()
    speedups, chars_per_step, avg_acc = [], [], []
    stype_frac = {"no_draft": [], "full_reject": [], "partial": [], "full_accept": []}
    mismatch_log: List[Dict] = []

    for toks, n_chars in targets:
        m = PlaybackMetrics()
        TensorSpeculativePlayback(tok, drafter, verifier, m).run_tokens(toks)
        steps = m.speculative_steps or 1
        speedups.append(m.speedup_ratio)
        chars_per_step.append(n_chars / steps)
        avg_acc.append(m.average_accepted_per_step)
        for st in stype_frac:
            stype_frac[st].append(m.step_types[st] / steps)
        if collect_mismatch and len(mismatch_log) < 12 and m.mismatch_log:
            mismatch_log.extend(m.mismatch_log[: 12 - len(mismatch_log)])

    return {
        "mean_speedup": float(np.mean(speedups)),
        "std_speedup": float(np.std(speedups)),
        "mean_chars_per_step": float(np.mean(chars_per_step)),
        "mean_avg_accepted": float(np.mean(avg_acc)),
        "step_types": {st: float(np.mean(v)) for st, v in stype_frac.items()},
        "mismatch_log": mismatch_log,
    }


# ── Per-tokenizer analysis ────────────────────────────────────────────────────
def analyse_tokenizer(
    name: str, repo: str, corpus_texts: List[str], target_texts: List[str],
    args,
) -> Dict:
    print("\n" + "#" * 78)
    print(f"# TOKENIZER: {name}  ({repo})")
    print("#" * 78)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

    # ── A. Static fertility / fragmentation ──
    sample = "\n".join(corpus_texts[:200])[: args.frag_sample_chars]
    fert = fertility_and_fragmentation(tok, sample, args.lang)
    print(f"\n[A] FERTILITY & FRAGMENTATION  (sample {fert['chars']:,} chars)")
    print(f"    tokens/char           : {fert['tokens_per_char']:.3f}  "
          f"(chars/token = {fert['chars_per_token']:.2f})")
    print(f"    tokens/byte           : {fert['tokens_per_byte']:.3f}")
    print(f"    byte-fragment rate    : {fert['byte_fragment_rate']:.1%}  "
          f"(tokens that decode to incomplete UTF-8)")
    print(f"    script char coverage  : {fert['script_single_char_coverage']:.1%}  "
          f"({fert['script_chars_tested']} codepoints tested)")
    print(f"    median token length   : {fert['median_token_chars']:.1f} chars")

    # ── Build corpus tokens ──
    print(f"\n[build] encoding corpus (target up to {args.max_corpus_tokens:,} tokens)...")
    corpus_tokens: List[int] = []
    for t in tqdm(corpus_texts, desc="  encode corpus", unit="art"):
        corpus_tokens.extend(encode(tok, t))
        if len(corpus_tokens) >= args.max_corpus_tokens:
            corpus_tokens = corpus_tokens[: args.max_corpus_tokens]
            break
    full_n = len(corpus_tokens)
    print(f"    corpus size: {full_n:,} tokens "
          f"({full_n / fert['tokens_per_char'] / 1e6:.1f}M chars of source)")

    # ── Encode targets (truncated) ──
    targets: List[Tuple[List[int], int]] = []
    for t in target_texts:
        ids = encode(tok, t)[: args.max_target_tokens]
        if len(ids) >= 8:
            targets.append((ids, len(tok.decode(ids))))
    print(f"    targets: {len(targets)} held-out articles "
          f"(≤{args.max_target_tokens} tokens each)")

    # ── Index (max_k=2 for n≤3 on the full corpus) ──
    print(f"[build] indexing corpus (max_k=2)...")
    index = NGramIndex(corpus_tokens, max_k=2, cap_positions=args.cap_positions)

    results: Dict = {"repo": repo, "fertility": fert, "corpus_tokens": full_n,
                     "n_targets": len(targets)}

    # ── B/C. Corpus-size sensitivity (n=3, depth budget B) ──
    sizes = [s for s in args.corpus_sizes if s <= full_n] + [full_n]
    sizes = sorted(set(sizes))
    print(f"\n[B/C] CORPUS-SIZE SENSITIVITY  (n=3, depth B={args.size_sweep_budget})")
    print(f"      {'corpus':>10} {'speedup':>9} {'chars/step':>11} {'no_draft':>9} {'full_acc':>9}")
    size_rows = []
    for s in sizes:
        drafter = IndexedTensorNGramDrafter(
            index, n=3, num_sequences=1, draft_depth=args.size_sweep_budget, size_limit=s)
        agg = run_targets(tok, drafter, targets)
        size_rows.append({"size": s, **agg})
        print(f"      {s:>10,} {agg['mean_speedup']:>8.3f}x "
              f"{agg['mean_chars_per_step']:>10.2f} "
              f"{agg['step_types']['no_draft']:>8.1%} "
              f"{agg['step_types']['full_accept']:>8.1%}")
    results["corpus_size_sweep"] = size_rows

    # ── B. Budget sweep (n=3, depth, full corpus) ──
    print(f"\n[B] BUDGET SWEEP  (n=3, depth, full corpus)")
    print(f"    {'B':>3} {'speedup':>9} {'avg_acc':>8} {'chars/step':>11} "
          f"{'no_draft':>9} {'partial':>8} {'full_acc':>9}")
    budget_rows = []
    for b in range(1, args.max_budget + 1):
        drafter = IndexedTensorNGramDrafter(index, n=3, num_sequences=1, draft_depth=b)
        agg = run_targets(tok, drafter, targets, collect_mismatch=(b == 3))
        budget_rows.append({"budget": b, **{k: v for k, v in agg.items() if k != "mismatch_log"}})
        if b == 3:
            results["mismatch_examples"] = agg["mismatch_log"]
        st = agg["step_types"]
        print(f"    {b:>3} {agg['mean_speedup']:>8.3f}x {agg['mean_avg_accepted']:>7.2f} "
              f"{agg['mean_chars_per_step']:>10.2f} {st['no_draft']:>8.1%} "
              f"{st['partial']:>7.1%} {st['full_accept']:>8.1%}")
    results["budget_sweep"] = budget_rows

    # ── D. Depth vs Width (full corpus, n=3) ──
    print(f"\n[D] DEPTH vs WIDTH  (n=3, full corpus, fixed budget B = S×T)")
    print(f"    {'budget':>6} {'mode':>14} {'S×T':>7} {'speedup':>9} {'chars/step':>11} {'avg_acc':>8}")
    dw_rows = []
    for B in args.depthwidth_budgets:
        configs = [("depth", 1, B)]
        if B >= 4:
            configs.append(("width-half", 2, B // 2))
        configs.append((f"width-full", B, 1))
        for mode, S, T in configs:
            drafter = IndexedTensorNGramDrafter(index, n=3, num_sequences=S, draft_depth=T)
            agg = run_targets(tok, drafter, targets)
            dw_rows.append({"budget": B, "mode": mode, "S": S, "T": T, **agg})
            print(f"    {B:>6} {mode:>14} {f'{S}x{T}':>7} {agg['mean_speedup']:>8.3f}x "
                  f"{agg['mean_chars_per_step']:>10.2f} {agg['mean_avg_accepted']:>7.2f}")
    results["depth_vs_width"] = dw_rows

    # ── N-gram size effect (n=2,3,4; smaller corpus slice, max_k=3) ──
    n_slice = min(full_n, args.neffect_corpus_tokens)
    print(f"\n[B] N-GRAM SIZE EFFECT  (depth B={args.size_sweep_budget}, corpus={n_slice:,} tok)")
    index3 = NGramIndex(corpus_tokens[:n_slice], max_k=3, cap_positions=args.cap_positions)
    n_rows = []
    for nval in (2, 3, 4):
        drafter = IndexedTensorNGramDrafter(
            index3, n=nval, num_sequences=1, draft_depth=args.size_sweep_budget)
        agg = run_targets(tok, drafter, targets)
        n_rows.append({"n": nval, **{k: v for k, v in agg.items() if k != "mismatch_log"}})
        print(f"    n={nval}: speedup={agg['mean_speedup']:.3f}x  "
              f"chars/step={agg['mean_chars_per_step']:.2f}  "
              f"no_draft={agg['step_types']['no_draft']:.1%}")
    results["ngram_effect"] = n_rows

    return results


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_charts(all_results: Dict, lang: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")
    names = list(all_results.keys())
    colors = {"qwen": "#7c3aed", "gemma": "#0ea5e9"}
    col = [colors.get(n, "#3b82f6") for n in names]

    # ── Figure 1: fertility | byte-fragment | speedup vs corpus size ──
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    fert_vals = [all_results[n]["fertility"]["tokens_per_char"] for n in names]
    ax[0].bar(names, fert_vals, color=col, alpha=0.85)
    for i, v in enumerate(fert_vals):
        ax[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    ax[0].set_title("Fertility on " + lang + " (tokens / char)\nlower = more efficient",
                    fontsize=11, fontweight="bold")
    ax[0].set_ylabel("tokens per character")

    frag_vals = [all_results[n]["fertility"]["byte_fragment_rate"] for n in names]
    ax[1].bar(names, frag_vals, color=col, alpha=0.85)
    for i, v in enumerate(frag_vals):
        ax[1].text(i, v, f"{v:.0%}", ha="center", va="bottom", fontweight="bold")
    ax[1].set_title("Byte-fragment rate\n(tokens that are incomplete UTF-8)",
                    fontsize=11, fontweight="bold")
    ax[1].set_ylabel("fraction of tokens")

    for n in names:
        rows = all_results[n]["corpus_size_sweep"]
        xs = [r["size"] / 1e6 for r in rows]
        ys = [r["mean_speedup"] for r in rows]
        ax[2].plot(xs, ys, marker="o", label=n, color=colors.get(n, "#3b82f6"), linewidth=2.3)
    ax[2].axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax[2].set_title("Speedup vs corpus size\n(n=3, depth)", fontsize=11, fontweight="bold")
    ax[2].set_xlabel("corpus size (M tokens)")
    ax[2].set_ylabel("mean speedup (x)")
    ax[2].legend()

    plt.suptitle(f"Wiki-{lang} Tokenizer RCA — Part 1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p1 = os.path.join(out_dir, f"wiki_{lang}_part1.png")
    plt.savefig(p1, dpi=200); plt.close()
    print(f"\nChart saved → {p1}")

    # ── Figure 2: step types | depth-vs-width | effective chars/step ──
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

    scolors = {"no_draft": "#94a3b8", "full_reject": "#ef4444",
               "partial": "#f59e0b", "full_accept": "#22c55e"}
    x = np.arange(len(names)); bottom = np.zeros(len(names))
    for st, sc in scolors.items():
        vals = [all_results[n]["budget_sweep"][2]["step_types"][st] for n in names]  # B=3
        ax[0].bar(x, vals, bottom=bottom, color=sc, label=st.replace("_", " "), width=0.55)
        bottom += np.array(vals)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_title("Step-type breakdown (n=3, B=3)", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("fraction of steps"); ax[0].legend(fontsize=8)

    width = 0.35
    budgets = sorted({r["budget"] for r in all_results[names[0]]["depth_vs_width"]})
    for j, n in enumerate(names):
        depth_sp = [next(r["mean_speedup"] for r in all_results[n]["depth_vs_width"]
                         if r["budget"] == b and r["mode"] == "depth") for b in budgets]
        wide_sp = [max(r["mean_speedup"] for r in all_results[n]["depth_vs_width"]
                       if r["budget"] == b and r["mode"].startswith("width")) for b in budgets]
        xb = np.arange(len(budgets)) + (j - 0.5) * width
        ax[1].bar(xb - 0.08, depth_sp, width=0.16, label=f"{n} depth", color=colors.get(n), alpha=0.9)
        ax[1].bar(xb + 0.08, wide_sp, width=0.16, label=f"{n} width(best)", color=colors.get(n), alpha=0.45)
    ax[1].set_xticks(np.arange(len(budgets))); ax[1].set_xticklabels([f"B={b}" for b in budgets])
    ax[1].set_title("Depth vs best-Width speedup\n(width helps the fragmented tokenizer more)",
                    fontsize=11, fontweight="bold")
    ax[1].set_ylabel("mean speedup (x)"); ax[1].legend(fontsize=7)

    cps = [all_results[n]["budget_sweep"][2]["mean_chars_per_step"] for n in names]  # B=3
    ax[2].bar(names, cps, color=col, alpha=0.85)
    for i, v in enumerate(cps):
        ax[2].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")
    ax[2].set_title("Effective rate: source chars / target step\n(fair cross-tokenizer, n=3 B=3)",
                    fontsize=11, fontweight="bold")
    ax[2].set_ylabel("characters per step")

    plt.suptitle(f"Wiki-{lang} Tokenizer RCA — Part 2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    p2 = os.path.join(out_dir, f"wiki_{lang}_part2.png")
    plt.savefig(p2, dpi=200); plt.close()
    print(f"Chart saved → {p2}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Low-resource tokenizer RCA via speculative decoding")
    ap.add_argument("--lang", default="lo", help="Wikipedia language code (lo/my/ar/ru/uk/...)")
    ap.add_argument("--date", default="20231101")
    ap.add_argument("--tokenizers", default="qwen,gemma", help="comma list from: " + ",".join(TOKENIZERS))
    ap.add_argument("--n-targets", type=int, default=300, dest="n_targets")
    ap.add_argument("--max-target-tokens", type=int, default=256, dest="max_target_tokens")
    ap.add_argument("--max-corpus-tokens", type=int, default=7_000_000, dest="max_corpus_tokens")
    ap.add_argument("--max-articles", type=int, default=None, dest="max_articles",
                    help="cap #articles loaded (for huge languages)")
    ap.add_argument("--corpus-sizes", default="250000,500000,1000000,2000000,4000000",
                    dest="corpus_sizes_str")
    ap.add_argument("--max-budget", type=int, default=6, dest="max_budget")
    ap.add_argument("--size-sweep-budget", type=int, default=4, dest="size_sweep_budget")
    ap.add_argument("--depthwidth-budgets", default="2,4,6", dest="depthwidth_budgets_str")
    ap.add_argument("--neffect-corpus-tokens", type=int, default=1_000_000, dest="neffect_corpus_tokens")
    ap.add_argument("--cap-positions", type=int, default=256, dest="cap_positions")
    ap.add_argument("--frag-sample-chars", type=int, default=200_000, dest="frag_sample_chars")
    args = ap.parse_args()
    args.corpus_sizes = [int(x) for x in args.corpus_sizes_str.split(",")]
    args.depthwidth_budgets = [int(x) for x in args.depthwidth_budgets_str.split(",")]

    out_dir = f"experiments/wiki_{args.lang}"
    art_dir = os.path.join(out_dir, "artifacts")
    os.makedirs(art_dir, exist_ok=True)

    print("=" * 78)
    print(f"LOW-RESOURCE TOKENIZER RCA — language: {args.lang}")
    print(f"tokenizers: {args.tokenizers} | targets: {args.n_targets} | "
          f"max corpus: {args.max_corpus_tokens:,} tok")
    print("=" * 78)

    # ── Load + split articles (corpus pool vs held-out targets) ──
    print(f"Loading Wikipedia ({args.date}.{args.lang})...")
    max_chars = None
    texts = load_articles(args.lang, date=args.date, max_chars=max_chars)
    if args.max_articles:
        texts = texts[: args.max_articles]
    rng = np.random.default_rng(42)
    order = rng.permutation(len(texts))
    target_idx = set(order[: args.n_targets].tolist())
    target_texts = [texts[i] for i in order[: args.n_targets]]
    corpus_texts = [texts[i] for i in order[args.n_targets:]]
    print(f"  {len(texts):,} articles  →  corpus pool {len(corpus_texts):,} | targets {len(target_texts):,}")

    all_results = {}
    for name in args.tokenizers.split(","):
        name = name.strip()
        repo = TOKENIZERS[name]
        all_results[name] = analyse_tokenizer(name, repo, corpus_texts, target_texts, args)

    # ── Cross-tokenizer summary ──
    print("\n" + "=" * 78)
    print("CROSS-TOKENIZER SUMMARY")
    print("=" * 78)
    print(f"{'tokenizer':>10} {'tok/char':>9} {'byte-frag':>10} {'cover':>7} "
          f"{'speedup':>9} {'chars/step':>11}")
    for n, r in all_results.items():
        f = r["fertility"]
        b3 = r["budget_sweep"][2]
        print(f"{n:>10} {f['tokens_per_char']:>9.3f} {f['byte_fragment_rate']:>9.1%} "
              f"{f['script_single_char_coverage']:>6.0%} {b3['mean_speedup']:>8.3f}x "
              f"{b3['mean_chars_per_step']:>10.2f}")

    # ── Save JSON + charts ──
    json_path = os.path.join(out_dir, "full_analysis.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"lang": args.lang, "args": vars(args), "results": all_results},
                  fh, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved → {json_path}")
    plot_charts(all_results, args.lang, art_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
