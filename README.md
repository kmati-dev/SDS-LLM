# Speculative Decoding Simulator (Greedy)

A **simulator** for greedy speculative decoding that **runs no real models** — it uses ground-truth tokens in place of the target model to measure speedup quickly on CPU. Use it to benchmark how well an n-gram drafter performs across different languages, datasets, and tokenizers.

Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Project structure

```text
├── pyproject.toml              # package + dev dependencies (pip install -e .[dev])
├── README.md
├── CONTRIBUTING.md
├── configs/
│   └── simulator_config.json   # default config for run_benchmark
├── src/specdecode/             # ── core package ──
│   ├── interface/              # Abstract base classes (drafter, verifier, playback + tensor variants)
│   ├── simulator/              # Concrete implementations
│   │   ├── drafter/            # NGramDrafter, TensorNGramDrafter, IndexedTensorNGramDrafter
│   │   ├── verifier/           # GreedyVerifier, TensorGreedyVerifier
│   │   ├── metrics/            # PlaybackMetrics
│   │   └── playback/           # SpeculativePlayback, TensorSpeculativePlayback
│   ├── datasets/               # Dataset loaders + REGISTRY (squad, xsum, samsum, cnn_dailymail, wiki[_demo])
│   └── analysis/               # RCA engine: DatasetAnalyzer + per-dataset hooks + wiki + summary
├── scripts/                    # ── entry points ──
│   ├── run_benchmark.py        # K-sweep benchmark per dataset
│   ├── analyze.py              # RCA: --dataset / --summary / --wiki
│   └── new_drafter.py          # scaffold a new drafter class
├── tests/                      # pytest suite
├── docs/
│   ├── architecture.md         # theory + design notes
│   ├── planning/               # planning docs
│   └── results/                # result writeups (deliverables)
└── experiments/                # per-dataset outputs (PNGs; large JSONs are gitignored)
```

---

## Installation

```bash
pip install -e .          # install specdecode package in editable mode (recommended)
# or without installing: prefix commands with PYTHONPATH=src
```

Then `import specdecode` or `from specdecode.simulator import NGramDrafter` works anywhere.

---

## Usage

### 1) Unit tests
```bash
python -m pytest tests/ -v
```

### 2) Benchmark (K-sweep) per dataset
```bash
python scripts/run_benchmark.py --dataset wiki_demo
python scripts/run_benchmark.py --dataset squad --tokenizer gpt2 --n 3 --max_draft 5
```
Output goes to `experiments/<dataset>/artifacts/` (results.json + speedup_benchmark.png)

### 3) Root-cause analysis
```bash
python scripts/analyze.py --dataset squad           # full RCA for one dataset (squad/xsum/samsum/cnn_dailymail)
python scripts/analyze.py --dataset xsum --limit 50 # quick run on first 50 samples
python scripts/analyze.py --summary                 # cross-dataset comparison table + combined charts
python scripts/analyze.py --wiki --lang lo          # low-resource tokenizer study
```

> Without `pip install -e .`, prefix with `PYTHONPATH=src`:
> `PYTHONPATH=src python scripts/analyze.py --dataset squad`

---

## Key components (`src/specdecode/`)

| Component | Classes | Location |
| :-- | :-- | :-- |
| Abstract contracts | `AbstractDrafter`, `AbstractVerifier`, `AbstractPlayback` (+ tensor variants) | [src/specdecode/interface/](src/specdecode/interface/) |
| N-gram drafter | `NGramDrafter`, `TensorNGramDrafter`, `IndexedTensorNGramDrafter` | [src/specdecode/simulator/drafter/](src/specdecode/simulator/drafter/) |
| Greedy verifier | `GreedyVerifier`, `TensorGreedyVerifier` | [src/specdecode/simulator/verifier/](src/specdecode/simulator/verifier/) |
| Playback loop | `SpeculativePlayback`, `TensorSpeculativePlayback` | [src/specdecode/simulator/playback/](src/specdecode/simulator/playback/) |
| Metrics + index | `PlaybackMetrics`, `NGramIndex` | [src/specdecode/simulator/metrics/](src/specdecode/simulator/metrics/) |
| RCA engine | `DatasetAnalyzer` + per-dataset subclasses | [src/specdecode/analysis/](src/specdecode/analysis/) |

Theory, speedup formula, and dependency injection design: [docs/architecture.md](docs/architecture.md)