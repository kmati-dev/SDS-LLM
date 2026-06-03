import os
import sys
import json
from transformers import AutoTokenizer
from datasets import load_dataset

# Set workspace paths for spec-decode-cnn-dailymail
SUBPROJECT_ROOT = "/Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/spec-decode-cnn-dailymail"
sys.path.insert(0, SUBPROJECT_ROOT)

from src.simulator import NGramDrafter, GreedyVerifier, PlaybackMetrics, SpeculativePlayback

def main():
    print("=" * 80)
    print("Running Speculative Decoding Mismatch Diagnostic for spec-decode-cnn-dailymail")
    print("=" * 80)
    
    # 1. Load config
    config_path = os.path.join(SUBPROJECT_ROOT, "configs", "simulator_config.json")
    tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            tokenizer_name = config.get("tokenizer_name", tokenizer_name)
            
    print(f"Loading tokenizer: {tokenizer_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading {tokenizer_name}, falling back to 'gpt2': {e}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 2. Load matched pair from CNN/DailyMail (row 42)
    print("Fetching CNN/DailyMail dataset row 42...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    sample = dataset[42]
    article_text = sample["article"]
    highlights_text = sample["highlights"]
    
    corpus_tokens = tokenizer.encode(article_text)
    target_tokens = tokenizer.encode(highlights_text)
    
    print(f"Article Tokens (Corpus):    {len(corpus_tokens)}")
    print(f"Highlights Tokens (Target): {len(target_tokens)}")
    print("-" * 80)
    
    strategies = ["first", "recency", "frequency"]
    results = {}
    
    for strategy in strategies:
        print(f"\nRunning simulation with matching_strategy='{strategy}'...")
        metrics = PlaybackMetrics()
        drafter = NGramDrafter(corpus_tokens=corpus_tokens, n=3, draft_size=3, matching_strategy=strategy)
        verifier = GreedyVerifier()
        
        playback = SpeculativePlayback(
            tokenizer=tokenizer,
            drafter=drafter,
            verifier=verifier,
            metrics=metrics
        )
        
        reconstructed = playback.run_playback(highlights_text, use_drafter=True)
        summary = metrics.get_summary()
        mismatches = metrics.mismatch_records
        
        results[strategy] = {
            "summary": summary,
            "mismatches": mismatches
        }
        
        print(f"[{strategy}] Steps: {summary['speculative_steps']} | Speedup: {summary['speedup_ratio']}x | Avg Accept: {summary['average_accepted_per_step']} | Total Mismatches logged: {len(mismatches)}")
        
        # Print top 5 mismatches in detail
        print(f"\n--- Detailed Analysis of Top 5 Mismatches for '{strategy}' (Subproject) ---")
        for idx, mismatch in enumerate(mismatches[:5]):
            step = mismatch["step_index"]
            context_ids = mismatch["prompt_context_ids"][-6:] # last 6 context tokens
            context_text = tokenizer.decode(context_ids)
            
            draft_ids = mismatch["draft_token_ids"]
            draft_text = tokenizer.decode(draft_ids) if draft_ids else "<NO DRAFT>"
            
            accepted_cnt = mismatch["accepted_count"]
            accepted_ids = draft_ids[:accepted_cnt]
            accepted_text = tokenizer.decode(accepted_ids) if accepted_ids else "<NONE>"
            
            expected_id = mismatch["expected_token_id"]
            expected_text = tokenizer.decode([expected_id]) if expected_id is not None else "<EOF>"
            
            mismatched_draft_id = mismatch["mismatched_draft_token_id"]
            mismatched_draft_text = tokenizer.decode([mismatched_draft_id]) if mismatched_draft_id is not None else "<NONE>"
            
            explanation = mismatch["explanation"]
            reason = explanation.get("reason", "No reason provided")
            n_used = explanation.get("n_used", "N/A")
            
            print(f"\nMismatch #{idx+1} at Step {step}:")
            print(f"  * Prompt Context (last 6 tokens): '{repr(context_text)}'")
            if draft_ids:
                print(f"  * Drafted Sequence:               '{repr(draft_text)}'")
                print(f"  * Accepted Draft Prefix:          '{repr(accepted_text)}'")
                print(f"  * Rejection Point: Expected '{repr(expected_text)}' | Drafted '{repr(mismatched_draft_text)}'")
            else:
                print(f"  * Drafted Sequence:               <NO DRAFT GENERATED (fallback)>")
                print(f"  * Next Expected Token:            '{repr(expected_text)}'")
            print(f"  * Explanation of Draft decision:  {reason} (N-gram size used: {n_used})")
            
    # Write subproject analysis summary file
    os.makedirs(os.path.join(SUBPROJECT_ROOT, "artifacts"), exist_ok=True)
    analysis_log_path = os.path.join(SUBPROJECT_ROOT, "artifacts", "mismatch_analysis_report.json")
    
    serializable_results = {}
    for strat, data in results.items():
        serializable_results[strat] = {
            "summary": data["summary"],
            "mismatch_count": len(data["mismatches"]),
            "sample_mismatches": data["mismatches"][:15]
        }
    
    with open(analysis_log_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed JSON report saved to '{analysis_log_path}'")

if __name__ == "__main__":
    main()
