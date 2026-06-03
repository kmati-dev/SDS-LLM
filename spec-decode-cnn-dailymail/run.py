import os
import json
import argparse
import random

import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset

from src.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback

# ---------------------------------------------------------------------------
# Module-level dataset cache — avoids reloading on every function call
# ---------------------------------------------------------------------------
_DATASET_CACHE = None


def _get_dataset():
    """Load the CNN/DailyMail dataset once and cache it at module level."""
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        print("Fetching CNN/DailyMail dataset from Hugging Face...")
        _DATASET_CACHE = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    return _DATASET_CACHE


def get_cnn_dailymail_pair(index: int = None) -> tuple:
    """Return (article, highlights) from the *same* row."""
    dataset = _get_dataset()

    if index is None:
        index = random.randint(0, len(dataset) - 1)

    sample = dataset[index]
    article = sample["article"]       
    highlights = sample["highlights"] 
    return article, highlights


def load_default_config() -> dict:
    """Load benchmark hyper-parameters from configs/simulator_config.json."""
    config_path = os.path.join(os.path.dirname(__file__), "configs", "simulator_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not load config ({config_path}): {e}")
    return {}


# ---------------------------------------------------------------------------
# Main Benchmark Execution
# ---------------------------------------------------------------------------
def run_benchmark(tokenizer_name: str, n_gram_size: int, max_draft: int, artifacts_dir: str):
    print("=" * 70)
    print("Starting Speculative Decoding Simulator Sweep")
    print(f"Tokenizer:      {tokenizer_name}")
    print(f"N-gram Size:    {n_gram_size}-gram")
    print(f"Max Draft:      K = {max_draft}")
    print(f"Artifacts Dir:  {artifacts_dir}")
    print("=" * 70)

    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Load tokenizer
    print("Loading HuggingFace tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading {tokenizer_name}, falling back to 'gpt2': {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Load a SINGLE matched pair from CNN/DailyMail
    article_text, summary_text = get_cnn_dailymail_pair(index=42)

    corpus_tokens = tokenizer.encode(article_text)   
    target_tokens = tokenizer.encode(summary_text)   

    print(f"Corpus Tokens  (Article  → Drafter knowledge): {len(corpus_tokens)}")
    print(f"Target Tokens  (Summary  → Generation target): {len(target_tokens)}")
    print("-" * 70)

    draft_sizes = list(range(1, max_draft + 1))
    speedups = []
    avg_accepted = []

    # 3. Baseline — normal token-by-token decoding
    baseline_metrics = PlaybackMetrics()
    baseline_playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=None,       # type: ignore
        verifier=GreedyVerifier(),
        metrics=baseline_metrics,
    )
    baseline_playback.run_playback(summary_text, use_drafter=False)
    baseline_steps = baseline_metrics.speculative_steps
    print(f"Baseline (Normal Decoding): {baseline_steps} steps, 1.0x Speedup")

    # 4. Speculative sweep K = 1 … max_draft
    for k in draft_sizes:
        metrics = PlaybackMetrics()
        drafter = NGramDrafter(corpus_tokens=corpus_tokens, n=n_gram_size, draft_size=k)
        verifier = GreedyVerifier()

        playback = SpeculativePlayback(
            tokenizer=tokenizer,
            drafter=drafter,
            verifier=verifier,
            metrics=metrics,
        )

        reconstructed = playback.run_playback(summary_text, use_drafter=True)
        result = metrics.get_summary()

        speedups.append(result["speedup_ratio"])
        avg_accepted.append(result["average_accepted_per_step"])

        print(
            f"Speculative (K={k}): "
            f"Steps = {result['speculative_steps']} | "
            f"Avg Accept = {result['average_accepted_per_step']} | "
            f"Speedup = {result['speedup_ratio']}x"
        )

        assert reconstructed == summary_text, (
            f"Reconstructed text does not match target at K={k}!"
        )

    print("-" * 70)
    print("Benchmark complete! Generating chart...")

    # 5. Plot results
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("ggplot")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    primary   = "#3b82f6"  
    secondary = "#8b5cf6"  

    ax1.plot(draft_sizes, speedups, marker="o", color=primary,
             linewidth=2.5, markersize=8, label="Speedup Ratio")
    ax1.axhline(1.0, color="#ef4444", linestyle="--", alpha=0.7, label="Baseline (1.0x)")
    ax1.set_title("Inference Acceleration (Speedup Ratio)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Speculative Draft Size (K)", fontsize=12, labelpad=10)
    ax1.set_ylabel("Speedup Multiplier (x)", fontsize=12, labelpad=10)
    ax1.set_xticks(draft_sizes)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

    ax2.bar(draft_sizes, avg_accepted, color=secondary, alpha=0.85,
            edgecolor="#6d28d9", width=0.5, label="Avg Accepted Tokens")
    ax2.set_title("Average Speculated Tokens Accepted per Step", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Speculative Draft Size (K)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Average Tokens Accepted", fontsize=12, labelpad=10)
    ax2.set_xticks(draft_sizes)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

    plt.suptitle(
        f"Speculative Decoding Performance Summary ({tokenizer_name})\n"
        f"Greedy verification · {n_gram_size}-gram drafter · CNN/DailyMail",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout()

    output_path = os.path.join(artifacts_dir, "speedup_benchmark.png")
    plt.savefig(output_path, dpi=300)
    print(f"Chart saved → '{output_path}'")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    defaults = load_default_config()

    parser = argparse.ArgumentParser(description="Speculative Decoding Simulator Sweep")
    parser.add_argument("--tokenizer",    type=str, default=defaults.get("tokenizer_name", "Qwen/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--n",            type=int, default=defaults.get("n_gram_size", 3))
    parser.add_argument("--max_draft",    type=int, default=defaults.get("max_draft", 6))
    parser.add_argument("--artifacts_dir",type=str, default=defaults.get("artifacts_dir", "artifacts"))

    args = parser.parse_args()
    run_benchmark(args.tokenizer, args.n, args.max_draft, args.artifacts_dir)