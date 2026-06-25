# System Architecture: Greedy Speculative Decoding Simulator

This document provides a detailed technical overview of the speculative decoding simulator. It explains the core theoretical principles, key software engineering patterns, and concrete components of the project.

---

## 1. Theoretical Background

### Autoregressive Decoding Bottleneck
Standard inference in Large Language Models (LLMs) is executed **autoregressively**, producing one token at a time:
$$x_{t} \sim P(X_t \mid x_{<t})$$

Each token generation requires a complete forward pass of the neural network. Because modern LLMs are massive (containing billions of parameters), transferring weights from High-Bandwidth Memory (HBM) to GPU SRAM for a single token generation makes standard decoding heavily **memory-bound**. Computation hardware remains underutilized.

### Speculative Decoding Heuristic
Speculative decoding breaks this memory bottleneck by introducing a **dual-model framework**:
1. **Drafter (Draft Model - Small & Fast):** Speculates a sequence of $K$ future token guesses at very low computational cost.
2. **Verifier (Target Model - Large & High Capacity):** Runs a *single, parallel forward pass* to verify all $K$ speculations in parallel.

If the verifier accepts $i$ draft tokens, we obtain $i+1$ tokens (including one recovery token) for the execution cost of a single target model step.

---

## 2. Core Simulator Components

The architecture follows a strict decoupled contract utilizing **Dependency Injection** (defined in [interfaces.py](../src/specdecode/interfaces.py)):

```
                       ┌─────────────────────────┐
                       │    AbstractPlayback     │
                       └────────────┬────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ AbstractDrafter  │     │ AbstractVerifier │     │ PlaybackMetrics  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### 1. Abstract Interfaces (`src/specdecode/interfaces.py`)
- **`AbstractDrafter`:** Dictates the speculative generation contract. Requires implementing `generate_draft(prompt: List[int]) -> List[int]`.
- **`AbstractVerifier`:** Dictates the verification loop. Requires implementing `verify(draft_tokens, current_prefix, complete_tokens) -> Dict[str, Any]`.
- **`AbstractPlayback`:** Declares the execution flow. Coordinates the main loop via `run_playback(input_data: str, use_drafter: bool) -> str`.

### 2. N-Gram Drafter (`NGramDrafter` in `src/specdecode/simulator.py`)
To avoid running a second neural network locally, the simulator implements an **N-Gram Drafter** that serves as a highly deterministic mock for a draft model:
* **Lookup database:** Stores a provided text corpus (`corpus_tokens`).
* **Backoff Heuristic:**
  1. Searches the corpus for the current prompt's trailing $(N-1)$-gram sequence.
  2. If found, it speculates the subsequent $K$ tokens.
  3. If no match is found, it backs off to search for an $(N-2)$-gram, repeating this down to a $1$-gram context.
  4. If absolutely no matches exist, it returns an empty speculation list `[]`.

### 3. Greedy Verifier (`GreedyVerifier` in `src/specdecode/simulator.py`)
Compares speculations against the ground-truth target sequence (`complete_tokens`) sequentially:
1. Iterates from index $i = 0$ to $K-1$.
2. If `draft_tokens[i] == complete_tokens[prefix_len + i]`, the token is **Accepted**.
3. Upon the first mismatch, all subsequent speculated tokens are **Rejected** immediately, breaking the loop.
4. **Recovery Token:** Appends the true ground-truth token at the mismatch point, guaranteeing forward progress.

### 4. Metrics Tracker (`PlaybackMetrics` in `src/specdecode/simulator.py`)
Collects and tracks the simulation's performance:
* **Accepted Tokens:** Number of speculations verified and accepted.
* **Rejected Tokens:** Number of speculations rejected.
* **Speedup Ratio:** Calculated as:
  $$\text{Speedup Ratio} = \frac{\text{Normal Autoregressive Steps}}{\text{Speculative Verification Steps}}$$

---

## 3. Class Design & Dependency Injection

The simulator runtime (`SpeculativePlayback`) relies on **Duck Typing** for external dependencies like Tokenizers (e.g. HuggingFace `AutoTokenizer` or Qwen tokenizers). This allows swapping the tokenizers transparently:

```python
class SpeculativePlayback(AbstractPlayback):
    def __init__(
        self,
        tokenizer: Any,
        drafter: AbstractDrafter,
        verifier: AbstractVerifier,
        metrics: Optional[PlaybackMetrics] = None
    ) -> None:
        super().__init__(tokenizer, drafter, verifier, metrics)
```

This design decouples testing mockups from the production-ready benchmark environment.
