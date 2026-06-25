# Root Cause Analysis — Greedy Speculative Decoding บนภาษาลาว (Lao Wikipedia)

**หัวข้อที่วิเคราะห์:** พฤติกรรมของ **greedy speculative decoding** (n-gram drafter + greedy verifier)
เมื่อใช้ **Lao Wikipedia** เป็น corpus สำหรับการทำนาย — drafter เดาถูก/ผิดเมื่อไร, accept ได้กี่ token,
speedup เท่าไร, และอะไรเป็นตัวขับ/ตัวฉุด

> **tokenizer (Qwen 3.5 / Gemma 4) เป็นเพียง "setting ที่ลองสลับ"** ตามโจทย์ ไม่ใช่สิ่งที่นำมาตัดสิน —
> เรารัน speculative decoding ตัวเดียวกันภายใต้ tokenizer สองตัว เพื่อดูว่าการเลือก tokenizer
> ทำให้พฤติกรรม spec-decode เปลี่ยนไปอย่างไร

> ไฟล์: กราฟ `experiments/wiki_lo/artifacts/wiki_lo_part{1,2}.png` · ข้อมูลดิบ `experiments/wiki_lo/full_analysis.json`
> โค้ด `analyze_wiki.py` · รันซ้ำ: `/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang lo`

---

## 0. Setup

| รายการ | ค่า |
|---|---|
| Corpus (n-gram DB ของ drafter) | Lao Wikipedia `wikimedia/wikipedia / 20231101.lo` — 4,714 บทความ |
| Target (จำลองการ generate) | 300 บทความ held-out (drafter ไม่เคยเห็น) |
| Drafter | n-gram + backoff (n→1), index แบบ hash |
| Verifier | greedy เทียบ ground truth; รองรับ **depth `[1,B]`** และ **width `[S,T]`** |
| Metric หลัก | **speedup** (= normal steps / spec steps), **acceptance**, **step types** |
| Setting ที่สลับ | tokenizer = `Qwen/Qwen3.5-4B` หรือ `google/gemma-4-31B-it` |

---

## 1. ผลรวม Speculative Decoding (หัวข้อหลัก)

**Speedup โดยรวม (n=3, draft budget B=3):** **~1.5–1.6×** (Qwen 1.51× · Gemma 1.56×)
→ spec-decode บนลาวได้ผลปานกลาง ไม่สูงเท่างาน extractive อย่าง SQuAD (1.72×) แต่ดีกว่า baseline ชัดเจน

### 1.1 Step-type breakdown — drafter พังตรงไหน (n=3, B=3)
| | no_draft | full_reject | partial | full_accept |
|---|---|---|---|---|
| (Qwen) | 0.4% | **71.3%** | 20.1% | 8.6% |
| (Gemma) | 0.7% | **70.7%** | 17.1% | 11.5% |

→ **ข้อค้นพบสำคัญที่สุด:** `no_draft` ต่ำมาก (<1%) แต่ `full_reject` สูง ~71%
แปลว่า drafter **"หา n-gram match เจอเกือบทุกครั้ง แต่เดา token ตัวถัดไปผิด"**
**คอขวดคือความแม่นของการทำนาย ไม่ใช่ coverage** — นี่คือ root cause หลักของ spec-decode บนภาษานี้

### 1.2 Draft budget sweep (n=3, depth) — เดิมพันยาวขึ้นช่วยไหม
| B | speedup (Qwen) | speedup (Gemma) | avg accepted/step |
|---|---|---|---|
| 1 | 1.284× | 1.291× | ~0.30 |
| 3 | 1.509× | 1.558× | ~0.55 |
| 6 | 1.637× | 1.712× | ~0.70 |

→ budget มากขึ้น speedup เพิ่ม (แบบ diminishing) — แต่ `full_accept` ลดลง (จาก ~29% ที่ B=1 → ~5% ที่ B=6)
และ `partial` เพิ่มขึ้น เพราะต้องเดาถูกติดกันหลายตัวยากขึ้น

### 1.3 N-gram size effect (depth B=4, corpus 1M) — context ยาวขึ้นช่วยมาก
| n | speedup (Qwen) | speedup (Gemma) |
|---|---|---|
| 2 | 1.179× | 1.181× |
| 3 | 1.568× | 1.627× |
| 4 | **1.881×** | **1.853×** |

→ n=4 ดีกว่า n=3 ~+20-30% — **context ยาวขึ้น = ทำนายแม่นขึ้นชัดเจน** เป็น lever ที่ได้ผลที่สุด

### 1.4 Depth vs Width drafting (budget B = S×T เท่ากัน)
| budget | depth (1×B) | best width (S=2) |
|---|---|---|
| B=4 (Qwen) | 1.573× | **1.666×** (+5.9%) |
| B=6 (Qwen) | 1.637× | **1.803×** (+10.1%) |
| B=4 (Gemma) | **1.633×** | 1.603× |
| B=6 (Gemma) | 1.712× | 1.729× |

→ **width-drafting (กระจายเป็น 2 สาย hedge ที่ branch point) ช่วยได้** — มากในกรณีที่จุดแตกแขนงไม่แน่นอน
(width-full `S×1` แย่กว่า depth; **2 สายคือจุดดีที่สุด**)

### 1.5 Corpus-size sensitivity — ข้อมูลเยอะขึ้นช่วยไหม
| corpus | speedup (Qwen) | speedup (Gemma) | no_draft |
|---|---|---|---|
| 250k | 1.563× | 1.611× | ~2% |
| 1M | 1.568× | 1.627× | ~1% |
| full | 1.573× | 1.633× | ~0.5% |

→ **เกือบแบนราบ** — corpus โต 20 เท่า speedup ขยับ ~1% เท่านั้น
n-gram **coverage อิ่มตัวตั้งแต่ 250k token** (backoff หาเจอเกือบเสมอ) → **corpus size ไม่ใช่คันโยก**

### 1.6 ข้อจำกัดของโจทย์ (corpus 1M/10M/100M)
ทั้งภาษาลาวบน Wikipedia มี token รวม ~4.9M (Qwen) / ~2.9M (Gemma) เท่านั้น
→ **corpus 10M/100M token สร้างไม่ได้จริง** สำหรับลาว (ทำได้แค่ 1M); และจาก §1.5 ต่อให้มีก็คงไม่ช่วย

---

## 2. ผลของการสลับ tokenizer (ส่วน "ลองใช้ tokenizer ต่าง ๆ")

รัน spec-decode ตัวเดิมภายใต้ tokenizer 2 ตัว → **token-based speedup ใกล้กัน (1.51× vs 1.56×)**
ดูเผิน ๆ เหมือนเลือก tokenizer ไหนก็พอกัน — **แต่ไม่ใช่** เพราะ 1 token ของแต่ละตัวแทนข้อความไม่เท่ากัน

| ตัวขับเบื้องหลัง | Qwen 3.5 | Gemma 4 |
|---|---|---|
| ความยาว token เฉลี่ย (chars/token) | 1.06 | 1.79 |
| byte-fragment rate (token ที่เป็นไบต์ไม่ครบตัวอักษร) | 26.4% | 0.2% |

**เพื่อเทียบให้ยุติธรรม** จึง normalize speedup เป็น **"อักษรลาวที่ generate ได้ต่อ 1 target-model step"**:

| (n=3, B=3) | Qwen 3.5 | Gemma 4 |
|---|---|---|
| token / step (speedup) | 1.51× | 1.56× |
| **chars / step** | **1.51** | **2.59** |

→ พอวัดเป็น "อักษรต่อ step" **spec-decode ภายใต้ Gemma เดินหน้าได้เร็วกว่า ~1.7 เท่า**
สาเหตุ: Qwen แตกอักษรลาวเป็นไบต์ → drafter ต้องเดาทีละไบต์ (ยาก + แต่ละ step ได้ข้อความนิดเดียว);
Gemma มี subword ลาวจริง → drafter เดาทีละ "คำ" (แต่ละ accept พ่วงข้อความมากกว่า)

---

## 3. Mismatch RCA เชิงคุณภาพ — ทำไม draft ถึงถูก reject (n=3, B=3)

**ภายใต้ Qwen — reject ระดับ byte/sub-character:**
```
ctx='\n\nຄວ'   expected='�'(byte!)  drafted='ເ'      ← เฉลยเองยังเป็น byte-fragment
ctx='\n\n'      expected='ຄ'         drafted='ร'(ไทย)  ← สับสนลาว↔ไทยผ่าน byte pattern
```
**ภายใต้ Gemma — reject ระดับ "คำ":**
```
ctx='\n\n'      expected='ຄວາມ'(คำเต็ม)  drafted='ราย'(คำไทย)
ctx='\n\nຄວາມ'  expected='ໝ'            drafted='ຫນ'  ← เสียงเดียวกัน เขียนต่างกัน
```
→ ยืนยัน §1.1: drafter หา match เจอแต่เดาผิด และทั้งสอง setting มี **Lao↔Thai confusion** (สคริปต์ใกล้กัน + คำปนใน corpus)

---

## 4. สรุป Root Cause (เกี่ยวกับ Speculative Decoding)

1. **คอขวด = "เดาผิด" ไม่ใช่ "หาไม่เจอ"** — `full_reject` ~71%, `no_draft` <1%
   n-gram backoff หา match ได้เสมอ แต่ continuation มักผิด → ขีดจำกัดคือความ predictable ของภาษาลาวภายใต้ tokenization
2. **Corpus size ไม่ใช่คันโยก** — speedup แบนราบ 0.25M–4.9M (coverage อิ่มเร็ว); และลาวมีข้อมูลไม่ถึง 10M token อยู่แล้ว
3. **คันโยกที่ได้ผลจริง:** (ก) **n ใหญ่ขึ้น** (n=4 ≈ +20-30%), (ข) **width-drafting** (S=2, +6-10% โดยเฉพาะตอน budget สูง)
4. **draft budget** ช่วยแบบ diminishing — B สูงได้ speedup เพิ่มแต่ full-accept ยากขึ้น
5. **การเลือก tokenizer เปลี่ยนผลจริงเมื่อวัดให้ถูก** — token-speedup เท่ากันแต่ chars/step ต่าง ~1.7 เท่า
   (เป็นบทเรียนเรื่อง **การเลือก metric** ของ spec-decode ในงานข้ามภาษา/ข้าม tokenizer)

**Bottom line:** บนภาษาลาว n-gram speculative decoding ได้ ~1.5–1.7× โดยถูกจำกัดด้วย "ความแม่นของการเดา"
(ไม่ใช่ปริมาณข้อมูล) — ปรับ n และใช้ width-drafting ช่วยได้ และต้องระวังวัด speedup เป็น "อักษรต่อ step"
เมื่อเทียบข้าม tokenizer ไม่งั้นจะสรุปผิด

---

## 5. รันซ้ำ / เปลี่ยนภาษา
```bash
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang lo            # ลาว (อันนี้)
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang my            # ภาษาอื่น เปลี่ยนแค่ code
/opt/miniconda3/bin/python3.13 analyze_wiki.py --lang lo --n-targets 50   # demo เร็ว
```
ผลลัพธ์ออกที่ `experiments/wiki_<lang>/` (JSON + กราฟ) และ print ครบทาง stdout
