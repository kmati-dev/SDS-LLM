# Speculative Decoding RCA — ภาษาลาว (Low-Resource) × Tokenizer ยุคใหม่

**คำถามวิจัย:** tokenizer ของโมเดลยุคใหม่ (Qwen 3.5 vs Gemma 4) จัดการ **ภาษาลาว** ได้ดีแค่ไหน
และจะ "วัด" ความดีนั้นยังไงให้ยุติธรรม — โดยใช้ **greedy speculative decoding (n-gram drafter)** เป็นเครื่องมือวัด

> ไฟล์ที่เกี่ยวข้อง
> - กราฟ: `experiments/wiki_lo/artifacts/wiki_lo_part1.png`, `wiki_lo_part2.png`
> - ข้อมูลดิบ: `experiments/wiki_lo/full_analysis.json`
> - โค้ด: `analyze_wiki.py`, `src/datasets/wiki.py`, `src/simulator.py` (`IndexedTensorNGramDrafter`, `TensorSpeculativePlayback`)
> - รันซ้ำ: `/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang lo`

---

## 0. Setup & Methodology

| รายการ | ค่า |
|---|---|
| Dataset | Lao Wikipedia (`wikimedia/wikipedia`, `20231101.lo`) |
| ขนาด | 5,014 บทความ · 5.54M ตัวอักษร · 73% เป็นอักษรลาว |
| แบ่งข้อมูล | corpus pool 4,714 บทความ (สร้าง n-gram DB) · target 300 บทความ held-out (จำลองการ generate) |
| Tokenizer | `Qwen/Qwen3.5-4B` (BBPE, vocab 248k) · `google/gemma-4-31B-it` (SentencePiece, vocab 262k) |
| Drafter | n-gram + backoff (n→1), index แบบ hash ให้ไหวระดับล้าน token |
| Verifier | greedy เทียบกับ ground truth, รองรับ **depth-draft `[1,B]`** และ **width-draft `[S,T]`** |

**หลักการ:** ถ้า tokenizer แทนภาษาลาวได้ดี → sequence สั้นลง + n-gram เดาต่อได้แม่น → drafter ถูก accept เยอะ → เร็วขึ้น
จึงใช้ speculative decoding เป็น "มาตรวัด" คุณภาพ tokenizer ทางอ้อม

---

## 1. Finding #0 — เงื่อนไข corpus 10M/100M token *เป็นไปไม่ได้* สำหรับภาษาลาว

ทั้งภาษาลาวบน Wikipedia มีข้อความเท่ากันคือ ~5.2M ตัวอักษร แต่พอ tokenize ได้ token ไม่เท่ากัน:

| | Qwen 3.5 | Gemma 4 |
|---|---|---|
| corpus ทั้งหมด (token) | **4.88M** | **2.92M** |

→ แม้แต่ corpus 10M token ก็ **สร้างไม่ได้** เพราะทั้งภาษามีไม่ถึง (Gemma ยิ่งน้อยเพราะแทนประหยัดกว่า)
นี่คือลักษณะ low-resource ที่ชัดเจน: **ข้อมูลไม่พอแม้แต่จะทดลองตามโจทย์** → จึง sweep corpus ภายในที่มีจริง (0.25M–full)

---

## 2. กราฟ Part 1 — Fertility · Byte-fragment · Corpus size

### 2.1 Fertility (tokens/char) — ความประหยัดในการแทนภาษาลาว
| | Qwen 3.5 | Gemma 4 |
|---|---|---|
| tokens / char | 0.943 | **0.558** |
| chars / token | 1.06 | **1.79** |
| tokens / byte | 0.364 | 0.215 |

→ **Gemma แทนลาวประหยัดกว่า ~1.7 เท่า** — Gemma 1 token ≈ 1.79 ตัวอักษร แต่ Qwen 1 token ≈ แค่ 1.06 ตัวอักษร

### 2.2 Byte-fragment rate — tokenizer "รู้จัก" ภาษาลาวจริงไหม
วัดสัดส่วน token ที่ decode เดี่ยว ๆ แล้วได้ `�` (ไบต์ UTF-8 ไม่ครบตัวอักษร = byte fallback)

| | Qwen 3.5 | Gemma 4 |
|---|---|---|
| byte-fragment rate | **26.4%** | **0.2%** |
| single Lao-char coverage (83 codepoint) | 37.3% | **68.7%** |

→ **Qwen แตกอักษรลาว ~1 ใน 4 ออกเป็นไบต์ดิบ** (ไม่มี subword ลาวใน vocab เลยถอยไปใช้ byte)
ส่วน Gemma แทบไม่เคย (0.2%) เพราะ vocab 262k ของมัน train มากับ multilingual จริง ๆ → "รู้จัก" คำลาว

### 2.3 Corpus-size sensitivity (n=3, depth B=4)
| corpus | Qwen speedup | Gemma speedup |
|---|---|---|
| 250k | 1.563x | 1.611x |
| 1M | 1.568x | 1.627x |
| 4M / full | 1.573x | 1.633x |

→ **เกือบแบนราบ** — เพิ่ม corpus 20 เท่า speedup ขยับแค่ ~1% และ `no_draft` ลดจาก ~2% → ~0.5%
แปลว่า n-gram **coverage อิ่มตัวตั้งแต่ 250k token แล้ว** (backoff หาเจอเกือบทุกครั้ง)
→ **corpus size ไม่ใช่ตัวแปรสำคัญ** การมีข้อมูล 10M/100M (ถ้ามี) ก็คงไม่ช่วย

---

## 3. กราฟ Part 2 — Step types · Depth vs Width · Effective rate

### 3.1 Step-type breakdown (n=3, B=3) — คอขวดอยู่ที่ไหน
| | no_draft | full_reject | partial | full_accept |
|---|---|---|---|---|
| Qwen | 0.4% | **71.3%** | 20.1% | 8.6% |
| Gemma | 0.7% | **70.7%** | 17.1% | 11.5% |

→ **`no_draft` ต่ำมาก (<1%) แต่ `full_reject` สูง ~71%** — drafter "หาเจอ" แทบทุกครั้งแต่ "เดาผิด"
**คอขวดคือความแม่นของการทำนาย ไม่ใช่ coverage** ← ข้อสรุปสำคัญที่สุดของ RCA

### 3.2 Depth vs Width drafting (budget B = S×T เท่ากัน)
| budget | mode | Qwen speedup | Gemma speedup |
|---|---|---|---|
| B=4 | depth (1×4) | 1.573x | **1.633x** |
| B=4 | **width-half (2×2)** | **1.666x** (+5.9%) | 1.603x (−1.8%) |
| B=6 | depth (1×6) | 1.637x | 1.712x |
| B=6 | **width-half (2×3)** | **1.803x** (+10.1%) | 1.729x (+1.0%) |

→ **สมมุติฐานเป็นจริง:** width-draft (กระจายเดิมพันหลายสาย hedge ที่ branch point) **ช่วย Qwen เยอะ (+6–10%)**
แต่ช่วย Gemma แทบไม่ขยับ
**เพราะ:** Qwen ทำงานระดับ byte → จุดแตกแขนงไม่แน่นอนมาก → เดิมพันหลายทางคุ้ม;
Gemma ทำงานระดับคำ → token มั่นใจอยู่แล้ว → เดิมพันยาวเส้นเดียว (depth) พอ
(หมายเหตุ: width-full `S×1` แย่กว่า depth ทั้งคู่ — เดิมพันสายละ 1 token สั้นไป; **2 สายคือจุดที่ดีที่สุด**)

### 3.3 ⭐ Effective rate — เมตริกที่ยุติธรรมข้าม tokenizer
"speedup" ปกติวัดเป็น **token/step** ซึ่ง **ไม่ยุติธรรม** เพราะ 1 token ของ Qwen กับ Gemma แทนข้อความไม่เท่ากัน
จึงวัด **อักษรลาวที่สร้างได้จริงต่อ 1 target-model step** (`total_chars / speculative_steps`)

| (n=3, B=3) | Qwen 3.5 | Gemma 4 |
|---|---|---|
| token-speedup | 1.509x | 1.558x | ← ดูเหมือน **เท่ากัน** |
| **chars / step (จริง)** | **1.51** | **2.59** | ← Gemma เร็วกว่า **~1.7 เท่า** |

→ **นี่คือ punchline:** ถ้าดูแค่ token-speedup จะสรุปผิดว่า "tokenizer ไหนก็พอกัน"
แต่เมตริกที่ยุติธรรมเผยว่า **Gemma decode ลาวได้เร็วกว่าจริง ~70%** เพราะแต่ละ token พ่วงข้อความมากกว่า

---

## 4. N-gram size effect (depth B=4, corpus 1M)
| n | Qwen speedup | Gemma speedup | Gemma chars/step |
|---|---|---|---|
| 2 | 1.179x | 1.181x | 1.99 |
| 3 | 1.568x | 1.627x | 2.70 |
| 4 | **1.881x** | **1.853x** | **3.07** |

→ context ยิ่งยาว (n ใหญ่) ยิ่งแม่นมาก (n=4 ดีกว่า n=3 ~+20-30%) — เป็นอีก lever ที่ช่วยได้จริงทั้งคู่

---

## 5. Qualitative RCA — mismatch เกิดที่ "ระดับ" ไหน (n=3, B=3)

**Qwen — ผิดระดับ byte / sub-character:**
```
ctx='\n\nຄວ'      expected='�'(byte!)  drafted='ເ'     ← เฉลยเองยังเป็น byte-fragment
ctx='\n\n'         expected='ຄ'        drafted='ร'(ไทย) ← สับสนลาว↔ไทยผ่าน byte pattern
ctx='ຄວາມໝ'       expected='າ'        drafted='ົ'      ← เดาสระ/วรรณยุกต์ผิดทีละตัว
```

**Gemma — ผิดระดับ "คำ":**
```
ctx='\n\n'         expected='ຄວາມ'(คำเต็ม)  drafted='ราย'(คำไทย) ← ตัดสินใจระดับคำ
ctx='\n\nຄວາມ'     expected='ໝ'            drafted='ຫນ'         ← ໝ vs ຫນ = เสียงเดียวกัน เขียนต่างกัน
```

→ ตอกย้ำ root cause: Qwen เดาทีละไบต์ (ยากโดยธรรมชาติ) ส่วน Gemma เดาทีละคำ (แต่ละครั้งคุ้มกว่า, ใช้จำนวนครั้งน้อยกว่า)
ทั้งคู่มี **Lao↔Thai confusion** เพราะสองภาษาใช้สคริปต์ใกล้กัน + มีคำปนใน corpus

---

## 6. สรุป Root Cause Analysis

1. **ตัวแปรหลัก = คุณภาพการแทนภาษาลาวของ tokenizer.**
   Gemma (SP 262k, multilingual) แทนลาวเป็น subword จริง → 0.2% byte-fallback, 1.79 chars/token, coverage 69%.
   Qwen (BBPE) ถอยไปใช้ byte → 26% byte-fallback, 1.06 chars/token, coverage 37%.

2. **คอขวดคือ "เดาผิด" ไม่ใช่ "หาไม่เจอ".**
   `full_reject` ~71%, `no_draft` <1% ทั้งคู่ → backoff หา match ได้เสมอ แต่ continuation มักผิด
   = ขีดจำกัดอยู่ที่ความ predictable ของ sequence ภายใต้ tokenization นั้น

3. **Corpus size ไม่ใช่คันโยก.** speedup แบนราบ 0.25M–4.9M; และลาวมี token รวมไม่ถึง 10M อยู่แล้ว
   → โจทย์ 10M/100M ทั้งทำไม่ได้และคงไม่ช่วย

4. **คันโยกที่ช่วยจริง:** (ก) n ใหญ่ขึ้น (n=4 ≈ +20-30%), (ข) width-drafting (S=2) ช่วย Qwen ที่แตกย่อย +6-10%
   แต่ทั้งสองอย่างก็ **ปิดช่องว่างของ tokenizer ไม่ได้**

5. **เมตริกสำคัญพอ ๆ กับผลลัพธ์.** token-speedup บอกว่า Qwen≈Gemma (1.51 vs 1.56) — *หลอกตา*
   chars/step (ยุติธรรม) เผยว่า **Gemma เร็วกว่าจริง ~1.7 เท่า (2.59 vs 1.51)**

**Bottom line:** สำหรับภาษา low-resource อย่างลาว **tokenizer คือทุกอย่าง** — Gemma 4 ชนะ Qwen 3.5 ขาด
และต้องวัดด้วย "อักษรต่อ step" ไม่ใช่ "token ต่อ step" ถึงจะเห็นความจริง

---

## 7. Reproducibility & Reusability (สำหรับเพื่อนในกลุ่ม)

ทุกอย่าง parameterize ด้วยภาษา — **เปลี่ยนแค่ `--lang` แล้วรันได้ทั้ง pipeline:**
```bash
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang lo   # ลาว (อันนี้)
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang my   # เมียนมา
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang ar   # อาหรับ
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang ru   # รัสเซีย
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang uk   # ยูเครน
```
ปรับได้: `--tokenizers qwen,gemma` · `--n-targets` · `--corpus-sizes` · `--max-budget` · `--depthwidth-budgets`
ผลลัพธ์ออกที่ `experiments/wiki_<lang>/` (JSON + กราฟ) และ print ครบทาง stdout
