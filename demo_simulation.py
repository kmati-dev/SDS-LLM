import os
import argparse
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback


def get_wiki_style_corpus() -> str:
    """Returns a rich technical corpus to train the N-gram Drafter."""
    return """
    Speculative decoding is a powerful technique designed to accelerate Large Language Model (LLM) inference. 
    Standard LLM decoding generates tokens one by one in an autoregressive fashion, which is heavily memory-bound. 
    Each forward pass of a large model is computationally expensive and slow because weights must be loaded from 
    high-bandwidth memory to the GPU SRAM for every single token generated.

    To solve this bottleneck, speculative decoding introduces a dual-model framework consisting of a small, fast 
    draft model (the Drafter) and a large, high-capacity target model (the Verifier). The fast draft model guesses 
    a sequence of K future tokens (speculative tokens) at a very low cost. 
    Then, the large target model runs a single parallel forward pass to verify all K speculative tokens in parallel.
    Since the target model verifies K tokens in a single step, if the drafts are accepted, we get K tokens for the 
    computational cost of just one target model step, leading to substantial speedups.

    In greedy speculative decoding, we compare the speculative tokens directly against the argmax predictions of the 
    target model. We accept tokens sequentially until the first mismatch occurs. If a mismatch is found at index i, 
    we reject all subsequent tokens, accept the matched tokens, and append a recovery token provided by the target model's 
    ground truth. This ensures that the final output matches the exact distribution of the target model perfectly 
    while saving computation steps.
    """


def get_test_target_text() -> str:
    """Returns a test prompt/ground truth text to simulate playback generation."""
    return """
    Speculative decoding is a powerful technique designed to accelerate Large Language Model inference.
    Each forward pass of a large model is computationally expensive.
    To solve this bottleneck, speculative decoding introduces a draft model and a large target model.
    The draft model guesses a sequence of speculative tokens at a low cost.
    Then, the large target model runs a single parallel forward pass to verify speculative tokens.
    In greedy speculative decoding, we accept tokens sequentially until the first mismatch.
    This ensures that the final output matches the exact distribution.
    """


def run_benchmark(tokenizer_name: str, n_gram_size: int, max_draft: int):
    print("=" * 70)
    print(f"Starting Speculative Decoding Simulator Sweep")
    print(f"Tokenizer:  {tokenizer_name}")
    print(f"N-gram Size: {n_gram_size}-gram")
    print(f"Max Draft:   K = {max_draft}")
    print("=" * 70)

    # 1. Load Tokenizer
    print("Loading HuggingFace tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading {tokenizer_name}, falling back to 'gpt2'...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 2. Tokenize Corpus and Target Text
    corpus_text = get_wiki_style_corpus()
    target_text = get_test_target_text()

    corpus_tokens = tokenizer.encode(corpus_text)
    target_tokens = tokenizer.encode(target_text)

    print(f"Corpus Tokens: {len(corpus_tokens)}")
    print(f"Target Tokens: {len(target_tokens)} to playback")
    print("-" * 70)

    draft_sizes = list(range(1, max_draft + 1))
    speedups = []
    avg_accepted = []
    step_counts = []

    # 3. Run non-speculative baseline (K=0)
    baseline_metrics = PlaybackMetrics()
    baseline_playback = SpeculativePlayback(
        tokenizer=tokenizer,
        drafter=None,  # type: ignore
        verifier=GreedyVerifier(),
        metrics=baseline_metrics
    )
    baseline_playback.run_playback(target_text, use_drafter=False)
    baseline_steps = baseline_metrics.speculative_steps
    print(f"Baseline (Normal Decoding): {baseline_steps} steps, 1.0x Speedup")

    # 4. Sweep Speculative Draft Sizes (K = 1 to max_draft)
    for k in draft_sizes:
        metrics = PlaybackMetrics()
        drafter = NGramDrafter(corpus_tokens=corpus_tokens, n=n_gram_size, draft_size=k)
        verifier = GreedyVerifier()
        
        playback = SpeculativePlayback(
            tokenizer=tokenizer,
            drafter=drafter,
            verifier=verifier,
            metrics=metrics
        )
        
        reconstructed = playback.run_playback(target_text, use_drafter=True)
        summary = metrics.get_summary()
        
        speedups.append(summary["speedup_ratio"])
        avg_accepted.append(summary["average_accepted_per_step"])
        step_counts.append(summary["speculative_steps"])

        print(
            f"Speculative (K={k}): "
            f"Steps = {summary['speculative_steps']} | "
            f"Avg Accept = {summary['average_accepted_per_step']} | "
            f"Speedup = {summary['speedup_ratio']}x"
        )
        assert reconstructed == target_text, "Error: Reconstructed text does not match target!"

    print("-" * 70)
    print("Benchmark complete! Generating premium statistics chart...")

    # 5. Plotting results with a premium, sleek aesthetic
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Vibrant custom colors
    primary_color = "#3b82f6"  # Premium royal blue
    secondary_color = "#8b5cf6"  # Soft indigo/purple
    accent_color = "#10b981"  # Emerald green

    # Left Plot: Speedup Ratio
    ax1.plot(draft_sizes, speedups, marker='o', color=primary_color, linewidth=2.5, markersize=8, label="Speedup Ratio")
    ax1.axhline(1.0, color="#ef4444", linestyle="--", alpha=0.7, label="Baseline (1.0x)")
    ax1.set_title("Inference Acceleration (Speedup Ratio)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Speculative Draft Size (K)", fontsize=12, labelpad=10)
    ax1.set_ylabel("Speedup Multiplier (x)", fontsize=12, labelpad=10)
    ax1.set_xticks(draft_sizes)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

    # Right Plot: Avg Accepted Tokens
    ax2.bar(draft_sizes, avg_accepted, color=secondary_color, alpha=0.85, edgecolor="#6d28d9", width=0.5, label="Avg Accepted Tokens")
    ax2.set_title("Average Speculated Tokens Accepted per Step", fontsize=14, fontweight="bold", pad=15)
    ax2.set_xlabel("Speculative Draft Size (K)", fontsize=12, labelpad=10)
    ax2.set_ylabel("Average Tokens Accepted", fontsize=12, labelpad=10)
    ax2.set_xticks(draft_sizes)
    ax2.grid(True, linestyle=":", alpha=0.6, axis='y')
    ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

    plt.suptitle(
        f"Speculative Decoding Performance Summary ({tokenizer_name})\n"
        f"Greedy verification with a {n_gram_size}-gram lookback drafter",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout()
    
    # Save the plot
    output_filename = "speedup_benchmark.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Chart saved successfully to '{output_filename}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speculate Decoding Simulator Sweep & Benchmark")
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct", 
        help="HuggingFace tokenizer name to load (default: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=3, 
        help="N-gram context size for the draft model (default: 3)"
    )
    parser.add_argument(
        "--max_draft", 
        type=int, 
        default=6, 
        help="Maximum draft size (K) to benchmark sweep (default: 6)"
    )

    args = parser.parse_args()
    run_benchmark(args.tokenizer, args.n, args.max_draft)
