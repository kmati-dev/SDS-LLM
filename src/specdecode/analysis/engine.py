"""
DatasetAnalyzer — the shared RCA engine for greedy speculative decoding.

A single full-dataset pass collects every metric the text report and both chart
figures need (speedup distribution, step-type breakdown per K, acceptance
histogram, n-gram-size effect, corpus-size sensitivity, mismatch examples), then
prints a report and renders two figures.

Per-dataset differences are isolated to a handful of hooks that subclasses
override — the raw source (``load_raw``/``extract``), the alignment metric
(``alignment_metric`` + ``alignment_label``; lexical overlap vs token novelty),
the dataset-specific deep dive (``deepdive``), and the prose ``key_findings``.
Everything else is identical across SQuAD / XSum / SAMSum / CNN-DailyMail.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm
from transformers import AutoTokenizer

from .common import lexical_overlap, run_sim, save_json

STEP_TYPES = ["no_draft", "full_reject", "partial", "full_accept"]
STEP_COLORS = {"no_draft": "#94a3b8", "full_reject": "#ef4444",
               "partial": "#f59e0b", "full_accept": "#22c55e"}
_PCT = mticker.FuncFormatter(lambda y, _: f"{y:.0%}")


class DatasetAnalyzer:
    # ── per-dataset configuration (override in subclasses) ──
    name: str = ""
    title: str = ""                       # human label, e.g. "SQuAD (Extractive)"
    color: str = "#3b82f6"
    marker: str = "o"
    fixed_k: int = 3
    fixed_n: int = 3
    max_draft: int = 6
    n_vals: Tuple[int, ...] = (1, 2, 3, 4)
    fracs: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0)
    scatter_max: int = 500
    ckpt_interval: int = 1000
    alignment_label: str = "Lexical Overlap"   # scatter x-axis label
    default_tokenizer: str = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, output_root: str = "."):
        self.output_root = output_root
        self.exp_dir = os.path.join(output_root, "experiments", self.name)
        self.artifacts = os.path.join(self.exp_dir, "artifacts")
        self.checkpoint = os.path.join(self.exp_dir, "full_analysis.json")
        self.config_path = os.path.join(self.exp_dir, "config.json")

    # ── hooks ──────────────────────────────────────────────────────────────────
    def load_raw(self):
        """Return the full raw HF dataset split to iterate over."""
        raise NotImplementedError

    def extract(self, sample) -> Tuple[str, str]:
        """Map one raw sample to ``(corpus_text, target_text)``."""
        raise NotImplementedError

    def alignment_metric(self, corpus_tokens: List[int], target_tokens: List[int]) -> float:
        """Per-sample corpus↔target alignment (default: lexical overlap)."""
        return lexical_overlap(corpus_tokens, target_tokens)

    def key_findings(self, data: Dict) -> List[str]:
        return []

    def deepdive(self, tokenizer, corpus_tokens: List[int], target_tokens: List[int]) -> None:
        """Optional dataset-specific single-sample analysis (prints to stdout)."""
        return None

    # ── tokenizer ───────────────────────────────────────────────────────────────
    def load_tokenizer(self):
        name = self.default_tokenizer
        if os.path.exists(self.config_path):
            with open(self.config_path, encoding="utf-8") as f:
                name = json.load(f).get("tokenizer_name", name)
        print(f"Tokenizer: {name}")
        return AutoTokenizer.from_pretrained(name), name

    # ── full-dataset pass (single pass, checkpointed) ────────────────────────────
    def compute_all_data(self, tokenizer, ds_raw, limit=None, resume=True) -> Dict:
        n_total = len(ds_raw) if limit is None else min(limit, len(ds_raw))

        speedups, zero_drafts, alignments = [], [], []
        k_stype_sums = {str(k): {st: 0.0 for st in STEP_TYPES}
                        for k in range(1, self.max_draft + 1)}
        n_speedup_sums = {str(n): 0.0 for n in self.n_vals}
        accepted_hist: Dict[str, int] = {}
        frac_speedup_sums = {str(f): 0.0 for f in self.fracs}
        valid_count = 0
        start_idx = 0

        if resume and os.path.exists(self.checkpoint):
            with open(self.checkpoint, encoding="utf-8") as f:
                ckpt = json.load(f)
            # Only resume a checkpoint written by this engine (generic schema).
            if "alignments" in ckpt:
                speedups = ckpt.get("speedups", [])
                zero_drafts = ckpt.get("zero_drafts", [])
                alignments = ckpt.get("alignments", [])
                k_stype_sums = ckpt.get("k_stype_sums", k_stype_sums)
                n_speedup_sums = ckpt.get("n_speedup_sums", n_speedup_sums)
                accepted_hist = ckpt.get("accepted_hist", {})
                frac_speedup_sums = ckpt.get("frac_speedup_sums", frac_speedup_sums)
                valid_count = ckpt.get("valid_count", 0)
                start_idx = ckpt.get("next_idx", 0)
                print(f"\n[FULL-DATASET PASS — resuming from idx={start_idx}/{n_total}]")
        if start_idx == 0:
            print(f"\n[FULL-DATASET PASS — {n_total} samples]")

        def snapshot(next_idx):
            return {
                "speedups": speedups, "zero_drafts": zero_drafts, "alignments": alignments,
                "k_stype_sums": k_stype_sums, "n_speedup_sums": n_speedup_sums,
                "accepted_hist": accepted_hist, "frac_speedup_sums": frac_speedup_sums,
                "valid_count": valid_count, "next_idx": next_idx,
            }

        for idx in tqdm(range(start_idx, n_total), desc="Computing", unit="sample",
                        initial=start_idx, total=n_total):
            corpus_text, target_text = self.extract(ds_raw[idx])
            ct = tokenizer.encode(corpus_text)
            tt = tokenizer.encode(target_text)
            if len(tt) < 2:
                continue
            valid_count += 1

            mi = run_sim(ct, target_text, tokenizer, k=self.fixed_k, n=self.fixed_n)
            speedups.append(mi.speedup_ratio)
            zero_drafts.append(mi.step_types["no_draft"] / (mi.speculative_steps or 1))
            alignments.append(self.alignment_metric(ct, tt))

            for cnt in mi.step_accepted_counts:
                accepted_hist[str(cnt)] = accepted_hist.get(str(cnt), 0) + 1

            for k in range(1, self.max_draft + 1):
                mk = run_sim(ct, target_text, tokenizer, k=k, n=self.fixed_n)
                total = mk.speculative_steps or 1
                for st in STEP_TYPES:
                    k_stype_sums[str(k)][st] += mk.step_types[st] / total

            for n in self.n_vals:
                mn = run_sim(ct, target_text, tokenizer, k=self.fixed_k, n=n)
                n_speedup_sums[str(n)] += mn.speedup_ratio

            for frac in self.fracs:
                trunc = max(2, int(len(ct) * frac))
                mf = run_sim(ct[:trunc], target_text, tokenizer, k=self.fixed_k, n=self.fixed_n)
                frac_speedup_sums[str(frac)] += mf.speedup_ratio

            if resume and (idx + 1) % self.ckpt_interval == 0:
                save_json(self.checkpoint, snapshot(idx + 1))

        if resume:
            save_json(self.checkpoint, snapshot(n_total))
            print(f"  Results saved → {self.checkpoint}")

        n = valid_count or 1
        return {
            "speedups": speedups,
            "zero_drafts": zero_drafts,
            "alignments": alignments,
            "k_stype_means": {k: {st: v / n for st, v in stypes.items()}
                              for k, stypes in k_stype_sums.items()},
            "n_speedup_means": {nk: v / n for nk, v in n_speedup_sums.items()},
            "accepted_hist": accepted_hist,
            "frac_speedup_means": {fk: v / n for fk, v in frac_speedup_sums.items()},
            "valid_count": valid_count,
        }

    # ── single-sample deep dive (text) ──────────────────────────────────────────
    def single_sample_analysis(self, tokenizer, corpus_text: str, target_text: str) -> None:
        ct = tokenizer.encode(corpus_text)
        tt = tokenizer.encode(target_text)

        print("\n" + "=" * 65)
        print(f"{self.title or self.name} — ROOT CAUSE ANALYSIS  (deep dive — sample #0)")
        print("=" * 65)
        print(f"\n[OVERVIEW]")
        print(f"  Corpus: {len(ct)} tokens  |  Target: {len(tt)} tokens  "
              f"|  {self.alignment_label}: {self.alignment_metric(ct, tt):.1%}")
        self.deepdive(tokenizer, ct, tt)

        print(f"\n[SINGLE SAMPLE — SWEEP K=1..{self.max_draft}]")
        print(f"  {'K':<4}{'steps':>7}{'speedup':>9}{'no_draft':>10}"
              f"{'full_rej':>10}{'partial':>9}{'full_acc':>10}{'avg_acc':>9}")
        print("  " + "-" * 72)
        for k in range(1, self.max_draft + 1):
            m = run_sim(ct, target_text, tokenizer, k=k, n=self.fixed_n)
            st, total = m.step_types, (m.speculative_steps or 1)
            print(f"  {k:<4}{m.speculative_steps:>7}{m.speedup_ratio:>8.2f}x"
                  f"{st['no_draft']/total:>9.0%}{st['full_reject']/total:>10.0%}"
                  f"{st['partial']/total:>9.0%}{st['full_accept']/total:>10.0%}"
                  f"{m.average_accepted_per_step:>9.2f}")

        m = run_sim(ct, target_text, tokenizer, k=self.fixed_k, n=self.fixed_n)
        print(f"\n[DEEP DIVE — K={self.fixed_k}, n={self.fixed_n}]")
        print(f"  Speedup: {m.speedup_ratio}x  |  N-gram usage: {m.n_gram_usage}")
        print(f"  Step types: {m.step_types}")

        print(f"\n[MISMATCH EXAMPLES — K={self.fixed_k}]")
        corpus_set = set(ct)
        if m.mismatch_log:
            for i, mm in enumerate(m.mismatch_log):
                ctx = repr(tokenizer.decode(mm["context_ids"]))
                exp = repr(tokenizer.decode([mm["expected_id"]])) if mm["expected_id"] else "'<EOF>'"
                drft = repr(tokenizer.decode([mm["drafted_id"]])) if mm["drafted_id"] else "'<NONE>'"
                extra = self._mismatch_note(mm, corpus_set)
                print(f"  #{i+1}  context: {ctx}")
                print(f"       expected: {exp}  |  drafted: {drft}  |  "
                      f"accepted_before: {mm['accepted_count']}  |  n_used: {mm['n_used']}{extra}")
        else:
            print("  No mismatches — all drafts fully accepted!")

    def _mismatch_note(self, mm, corpus_set) -> str:
        """Hook: extra annotation per mismatch (e.g. paraphrase vs novel word)."""
        return ""

    # ── multi-sample stats (text) ────────────────────────────────────────────────
    def print_multi_sample_stats(self, data: Dict) -> None:
        sp, al, zd, n = data["speedups"], data["alignments"], data["zero_drafts"], data["valid_count"]
        print(f"\n[MULTI-SAMPLE ROBUSTNESS — {n:,} samples, K={self.fixed_k}]")
        print(f"  Speedup   : mean={np.mean(sp):.2f}x  std={np.std(sp):.2f}  "
              f"min={min(sp):.2f}x  max={max(sp):.2f}x")
        print(f"  {self.alignment_label:<10}: mean={np.mean(al):.1%}  std={np.std(al):.1%}")
        print(f"  Zero-draft: mean={np.mean(zd):.1%}  std={np.std(zd):.1%}")
        print(f"\n[N-GRAM SIZE EFFECT — K={self.fixed_k}, averaged over {n:,} samples]")
        for nk, s in sorted(data["n_speedup_means"].items(), key=lambda x: int(x[0])):
            print(f"  n={nk}: mean_speedup={s:.3f}x")
        findings = self.key_findings(data)
        if findings:
            print(f"\n[KEY FINDINGS]")
            for i, line in enumerate(findings, 1):
                print(f"  {i}. {line}")

    # ── charts ───────────────────────────────────────────────────────────────────
    def plot_charts(self, data: Dict) -> None:
        os.makedirs(self.artifacts, exist_ok=True)
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            plt.style.use("ggplot")
        n_samples = data["valid_count"]
        ks = list(range(1, self.max_draft + 1))

        # ── Figure 1: step types per K | acceptance histogram | n-gram effect ──
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        bottoms = [0.0] * len(ks)
        for st, sc in STEP_COLORS.items():
            vals = [data["k_stype_means"][str(k)][st] for k in ks]
            ax.bar(ks, vals, bottom=bottoms, color=sc, alpha=0.85,
                   label=st.replace("_", " ").title(), width=0.6)
            bottoms = [b + h for b, h in zip(bottoms, vals)]
        ax.set_title(f"Step Type Breakdown per K\n(mean over {n_samples:,} samples)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Draft Size K"); ax.set_ylabel("Mean % of steps")
        ax.set_xticks(ks); ax.yaxis.set_major_formatter(_PCT); ax.legend(fontsize=8)

        ax = axes[1]
        hist = data["accepted_hist"]
        total_steps = sum(hist.values()) or 1
        xs = list(range(0, self.fixed_k + 1))
        ax.bar(xs, [hist.get(str(x), 0) / total_steps for x in xs],
               color=self.color, alpha=0.8, edgecolor="white", width=0.6)
        ax.set_title(f"Accepted Tokens per Step (K={self.fixed_k})\n(pooled over {n_samples:,} samples)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Tokens accepted in one step"); ax.set_ylabel("Proportion of steps")
        ax.yaxis.set_major_formatter(_PCT); ax.set_xticks(xs)

        ax = axes[2]
        sp_n = [data["n_speedup_means"][str(n)] for n in self.n_vals]
        ax.plot(self.n_vals, sp_n, marker=self.marker, color=self.color, linewidth=2.5, markersize=8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
        for n, s in zip(self.n_vals, sp_n):
            ax.annotate(f"{s:.2f}x", (n, s), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_title(f"Mean Speedup vs N-gram Size n (K={self.fixed_k})\n(mean over {n_samples:,} samples)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("N-gram size (n)"); ax.set_ylabel("Mean Speedup (x)")
        ax.set_xticks(self.n_vals); ax.legend(fontsize=9)

        plt.suptitle(f"{self.title or self.name} — Part 1 (full dataset)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        p1 = os.path.join(self.artifacts, f"{self.name}_part1.png")
        plt.savefig(p1, dpi=300); plt.close()
        print(f"\nChart saved → {p1}")

        # ── Figure 2: speedup distribution | alignment scatter | corpus sensitivity ──
        sp, al = data["speedups"], data["alignments"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax = axes[0]
        ax.bar(range(len(sp)), sorted(sp, reverse=True), color=self.color, alpha=0.8, width=1.0)
        ax.axhline(np.mean(sp), color="#1e293b", linestyle="--", linewidth=2,
                   label=f"Mean = {np.mean(sp):.2f}x")
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Baseline")
        ax.set_title(f"Speedup Across {n_samples:,} Samples", fontsize=12, fontweight="bold")
        ax.set_xlabel("Sample (sorted by speedup)"); ax.set_ylabel("Speedup (x)"); ax.legend(fontsize=9)

        ax = axes[1]
        rng = np.random.default_rng(42)
        n_scatter = min(self.scatter_max, len(al))
        if n_scatter:
            sidx = rng.choice(len(al), size=n_scatter, replace=False)
            ax.scatter([al[i] for i in sidx], [sp[i] for i in sidx], color=self.color,
                       s=30, alpha=0.5, edgecolors="none", marker=self.marker)
        ax.set_title(f"Speedup vs {self.alignment_label}\n({n_scatter} sampled from {n_samples:,})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(self.alignment_label); ax.set_ylabel("Speedup (x)")
        ax.xaxis.set_major_formatter(_PCT); ax.grid(True, linestyle=":", alpha=0.6)

        ax = axes[2]
        sp_fracs = [data["frac_speedup_means"][str(f)] for f in self.fracs]
        ax.plot([f * 100 for f in self.fracs], sp_fracs, marker=self.marker, color=self.color,
                linewidth=2.5, markersize=8)
        for f, s in zip(self.fracs, sp_fracs):
            ax.annotate(f"{s:.2f}x", (f * 100, s), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Baseline")
        ax.set_title(f"Corpus Size Sensitivity\n(mean over {n_samples:,} samples)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Corpus size (% of full)"); ax.set_ylabel("Mean Speedup (x)"); ax.legend(fontsize=9)

        plt.suptitle(f"{self.title or self.name} — Part 2 (full dataset)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        p2 = os.path.join(self.artifacts, f"{self.name}_part2.png")
        plt.savefig(p2, dpi=300); plt.close()
        print(f"Chart saved → {p2}")

    # ── orchestration ────────────────────────────────────────────────────────────
    def run(self, limit=None) -> Dict:
        tokenizer, _ = self.load_tokenizer()
        ds_raw = self.load_raw()
        corpus_text, target_text = self.extract(ds_raw[0])
        data = self.compute_all_data(tokenizer, ds_raw, limit=limit, resume=(limit is None))
        self.single_sample_analysis(tokenizer, corpus_text, target_text)
        self.print_multi_sample_stats(data)
        self.plot_charts(data)
        return data
