# Plan — RCA: Tokenizer ยุคใหม่ × ภาษา Low-Resource (ลาว) × Depth-vs-Width Drafting

> ทำบน branch **`feat/tensor`** (drafter/verifier เป็น tensor แล้ว)
> reframe: ใช้ **greedy speculative decoding (n-gram drafter)** เป็น "เครื่องมือวัด"
> ว่า tokenizer ของ **Qwen3.5 vs Gemma 4** จัดการ **ภาษาลาว** ได้ดีแค่ไหน
> และ **depth-draft vs width-draft** อันไหนเหมาะกับภาษา low-resource มากกว่า

---

## 1. Context — ทำไมต้องทำ

คำสั่งกลุ่ม: ใช้ Wikipedia ของภาษา low-resource เป็น corpus ของ n-gram drafter,
ทดสอบกับ tokenizer ของโมเดลปัจจุบัน, ดูที่ขนาด corpus 1M/10M/100M token. ผมได้ส่วน **Lao wiki**.

คำถามจริง = **"tokenizer ตัวไหนเข้าใจภาษาลาวดีกว่า และวัดยังไงให้ยุติธรรม"**
speculative decoding ตอบได้: tokenizer ที่แทนลาวดี → sequence สั้น + n-gram เดาได้ → accept เยอะ → เร็ว.
branch `feat/tensor` เพิ่มมิติ **depth vs width** ให้ลองด้วย.

**ข้อกำหนดเพิ่ม (จาก user):** โค้ดต้อง reusable — เพื่อนในกลุ่มเปลี่ยนแค่ **ภาษา** (`--lang lo` → `--lang my/ar/ru/uk`) แล้วรันได้เลย.

## 2. ข้อเท็จจริงที่วัดมาแล้ว (กำหนด scope จริง)

| รายการ | ค่า |
|---|---|
| Lao wiki (`wikimedia/wikipedia`, `20231101.lo`) | 5,014 บทความ |
| total chars / UTF-8 bytes | 5.54M / 14.26M |
| สัดส่วนอักษรลาว | 73% |
| fertility Qwen2.5 บนลาว | ~1.18 tok/char (อังกฤษ ~0.27) |
| **ทั้ง corpus ≈ tokens** | **~6.5M** |
| ถึง 1M / 10M / 100M token | ✅ / ❌ / ❌ |

**Finding #0:** ทั้งภาษาลาวมี ~6.5M token → เงื่อนไข 10M/100M **ทำไม่ได้** = low-resource ขาดข้อมูลแม้แต่จะสร้าง corpus.
→ แกน corpus size เปลี่ยนเป็น **sweep ภายในที่มีจริง: 0.25M · 0.5M · 1M · 2M · 4M · full(~6.5M)**

### Preview finding (วัดบนสตริงลาว 23 ตัวอักษรเดียวกัน)
| Tokenizer | tokens | tok/char |
|---|---|---|
| **Gemma 4** (`google/gemma-4-31B-it`, vocab 262k) | 15 | **0.65** 🥇 |
| **Qwen 3.5** (`Qwen/Qwen3.5-4B`, vocab 248k) | 25 | 1.09 |
| _(DeepSeek V4 — ไม่ใช้แล้ว)_ | 40 | 1.74 |

→ Gemma แทนลาวประหยัดกว่า Qwen ~1.7 เท่า — **สมมุติฐานหลัก** ของงาน

## 3. Environment & Tokenizers (final)

- รันด้วย `/opt/miniconda3/bin/python3.13` — transformers 5.10.2, datasets 5.0.0, **torch 2.12.1**, matplotlib/numpy/tqdm ✓
- repo `.venv` (Py 3.14) มีแค่ torch — **ห้ามใช้**
- Tokenizer = **Qwen + Gemma เท่านั้น** (ตามที่ user เคาะ); ทั้งคู่โหลดได้ **ไม่ต้องมี HF token** (Gemma 4 ไม่ gated)
  - `Qwen/Qwen3.5-4B`, `google/gemma-4-31B-it` (โหลดเฉพาะไฟล์ tokenizer ไม่กี่ MB)

## 4. สิ่งที่จะวิเคราะห์ (Analysis Dimensions)

### A. Tokenizer fertility & fragmentation — *static*
1. **Fertility**: tokens/char, tokens/byte (Qwen vs Gemma บนลาว)
2. **Sub-character / byte-fragment rate**: สัดส่วน token ที่ decode เดี่ยว ๆ แล้วได้ `�`/ไม่ครบ char → วัด "byte fallback" ตรง ๆ
3. **Lao single-char coverage**: อักษรลาว ~80 codepoint ตัวไหนเป็น 1 token (รู้จัก) vs ถูกแตก
4. **Token-length distribution** (กี่ char ต่อ token)

### B. N-gram predictability ผ่านแต่ละ tokenizer — *speculative lens*
5. **Speedup** (tensor playback): corpus = Lao wiki tokens, target = บทความ held-out; sweep budget **B=1..6**, **n=2..4**
6. **Step-type breakdown**: no_draft / full_reject / partial / full_accept
7. **N-gram coverage / hit-rate**: 1 − no_draft = สัดส่วน context ที่เจอใน corpus
8. **Corpus-size sensitivity**: 0.25M→full — speedup ยังไต่ขึ้นไหม (low-resource คาดว่ายังไม่อิ่ม)

### C. ⭐ Depth vs Width drafting (axis ใหม่จาก branch tensor)
9. budget B เท่ากัน เทียบ **depth (S=1,T=B)** กับ **width (S=k,T=B/k)**:
   - speedup, accepted/step, chosen_sequence distribution (สาย width ไหนชนะบ่อย)
   - **สมมุติฐาน:** tokenizer ที่แตกย่อยลาวเยอะ (Qwen) → branch point ไม่แน่นอน → **width ช่วยมากกว่า**;
     Gemma (token ยาว แน่นอนกว่า) → depth พอ

### D. Cross-tokenizer fair comparison — *หัวใจ*
10. ⭐ **Effective generation rate = อักษรลาวที่สร้างได้ต่อ 1 target-model step** (`total_chars / speculative_steps`)
    → เทียบข้าม tokenizer ได้ยุติธรรม เพราะหักล้าง confound เรื่อง fertility
    (speedup tokens/step สูงไม่ได้แปลว่าเร็วจริง ถ้าแต่ละ token แทนข้อความนิดเดียว)
11. ตารางสรุป side-by-side: fertility · byte-frag% · speedup(depth/width) · chars/step

### E. Qualitative / mismatch RCA
12. decode context/expected/drafted ดู drafter พังตรงไหน (byte-boundary, คำหายาก, ตัวเลข, คำยืม)

## 5. งานวิศวกรรมที่ต้องเขียน

1. **`NGramIndex`** (ใหม่) — สร้าง index `k-gram → positions` ครั้งเดียวต่อ corpus (k=1..n−1), รองรับ `size_limit` สำหรับ corpus-size sweep
2. **`IndexedTensorNGramDrafter`** (ใหม่, `AbstractTensorDrafter`) — semantics เดียวกับ `TensorNGramDrafter` (depth/width/backoff/dedupe/pad) แต่ lookup ผ่าน index → ไหวระดับล้าน token
3. **`TensorSpeculativePlayback`** (ใหม่, `AbstractPlayback`) — loop ที่ใช้ tensor drafter+verifier, reuse `PlaybackMetrics` (draft_size = ความยาวจริงของ row ที่ถูกเลือก) — *ของเดิมไม่มี (PLAN_tensor_drafter ระบุ out of scope)*
4. **Loader** `src/datasets/wiki.py` — `load(lang, ...)` generic ทุกภาษา + แบ่ง corpus/target ระดับบทความ; register ใน `src/datasets/__init__.py`
5. **`analyze_wiki.py --lang lo`** — วน tokenizer × corpus-size × (B,n) × {depth,width}, เก็บ metric A–E, **print ออกจอ**, เซฟ `experiments/wiki_lo/full_analysis.json` + charts
6. **`docs/presentation_wiki_lao.md`** — writeup RCA สำหรับ present (deliverable หลัก)

> ของเดิมทั้งหมด (`NGramDrafter`, `GreedyVerifier`, `TensorNGramDrafter`, `TensorGreedyVerifier`, test 33+ เคส) **ไม่แตะ** — เพิ่มอย่างเดียว

## 6. Verification

- `IndexedTensorNGramDrafter` ต้องให้ผล **เท่ากับ** `TensorNGramDrafter` บน corpus เล็ก (regression)
- `pytest tests/ -v` เดิมต้องเขียวหมด
- `TensorSpeculativePlayback` ต้อง reconstruct target ครบ 100% ทุก config
- เริ่ม **Qwen + subset เล็ก** ให้ครบ pipeline → ขยายเต็ม corpus + Gemma
- รันด้วย `/opt/miniconda3/bin/python3.13`
