# Speculative Decoding Simulator (Greedy)

ตัว**จำลอง** greedy speculative decoding ที่ **ไม่รันโมเดลจริง** — ใช้ ground-truth tokens แทน
target model เพื่อวัด speedup เร็ว ๆ บน CPU แล้วใช้เป็น "เครื่องมือวัด" ว่าภาษา/dataset/tokenizer
แบบไหนทำให้ n-gram drafter เดาถูกบ่อยแค่ไหน

---

## โครงสร้างโปรเจกต์

```text
├── pyproject.toml              # package + pytest config (installable: pip install -e .)
├── README.md
├── configs/
│   └── simulator_config.json   # ค่า default ของ run_benchmark
├── src/specdecode/             # ── ตัว package หลัก ──
│   ├── interfaces.py           # Abstract contracts (list-based + tensor-based)
│   ├── simulator.py            # Drafter / Verifier / Playback / Metrics / NGramIndex
│   ├── datasets/               # dataset loaders + REGISTRY (squad, xsum, samsum, cnn_dailymail, wiki[_demo])
│   └── analysis/               # RCA engine: DatasetAnalyzer + per-dataset hooks + wiki + summary
├── scripts/                    # ── entry points (CLI บาง ๆ) ──
│   ├── run_benchmark.py        # benchmark K-sweep ต่อ dataset
│   └── analyze.py              # RCA แบบรวม: --dataset / --summary / --wiki
├── tests/                      # pytest (interfaces + tensor interfaces)
├── docs/
│   ├── architecture.md         # ทฤษฎี + ดีไซน์
│   ├── planning/               # บันทึกการวางแผน (tensor drafter, wiki_lao)
│   └── results/                # writeup ผลลัพธ์สำหรับนำเสนอ (deliverable)
└── experiments/                # ผลลัพธ์ต่อ dataset (PNG track, full_analysis.json ใหญ่ถูก gitignore)
```

---

## การติดตั้ง

```bash
pip install -e .          # ติดตั้ง package `specdecode` แบบ editable (แนะนำ)
# หรือไม่ติดตั้ง แล้วชี้ PYTHONPATH=src เวลารัน script
```

แล้ว `import specdecode` หรือ `from specdecode.simulator import NGramDrafter` ใช้ได้ทุกที่

---

## การใช้งาน

### 1) Unit tests
```bash
python -m pytest tests/ -v
```

### 2) Benchmark (K-sweep) ต่อ dataset
```bash
python scripts/run_benchmark.py --dataset wiki_demo
python scripts/run_benchmark.py --dataset squad --tokenizer gpt2 --n 3 --max_draft 5
```
ผลออกที่ `experiments/<dataset>/artifacts/` (results.json + speedup_benchmark.png)

### 3) Root-Cause Analysis (รวมเป็น CLI เดียว)
```bash
python scripts/analyze.py --dataset squad          # RCA เต็ม dataset เดียว (squad/xsum/samsum/cnn_dailymail)
python scripts/analyze.py --dataset xsum --limit 50 # quick run บน 50 sample แรก
python scripts/analyze.py --summary                 # ตารางเทียบข้าม dataset + กราฟรวม
python scripts/analyze.py --wiki --lang lo          # งานศึกษา tokenizer ภาษา low-resource
```

> ถ้าไม่ได้ `pip install -e .` ให้นำหน้าด้วย `PYTHONPATH=src` เช่น
> `PYTHONPATH=src python scripts/analyze.py --dataset squad`

---

## คอมโพเนนต์หลัก (ใน `src/specdecode/`)

| คอมโพเนนต์ | คลาส | ไฟล์ |
| :-- | :-- | :-- |
| Abstract contracts | `AbstractDrafter`, `AbstractVerifier`, `AbstractPlayback` (+ tensor variants) | [src/specdecode/interfaces.py](src/specdecode/interfaces.py) |
| N-gram drafter (+ backoff) | `NGramDrafter`, `TensorNGramDrafter`, `IndexedTensorNGramDrafter` | [src/specdecode/simulator.py](src/specdecode/simulator.py) |
| Greedy verifier | `GreedyVerifier`, `TensorGreedyVerifier` | [src/specdecode/simulator.py](src/specdecode/simulator.py) |
| Playback loop | `SpeculativePlayback`, `TensorSpeculativePlayback` | [src/specdecode/simulator.py](src/specdecode/simulator.py) |
| Metrics + index | `PlaybackMetrics`, `NGramIndex` | [src/specdecode/simulator.py](src/specdecode/simulator.py) |
| RCA engine | `DatasetAnalyzer` + subclass ต่อ dataset | [src/specdecode/analysis/](src/specdecode/analysis/) |

รายละเอียดทฤษฎี (speculative decoding heuristic, สูตร speedup, dependency injection) อยู่ที่
[docs/architecture.md](docs/architecture.md)
