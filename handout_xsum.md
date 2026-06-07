# เอกสารประกอบการนำเสนอ
## Speculative Decoding — กรณีศึกษา: XSum Dataset

---

## 1. ภาพรวม

งานนี้ศึกษาการจำลอง **Speculative Decoding** โดยใช้ N-gram Drafter กับ dataset ประเภท Abstractive Summarization ชื่อว่า **XSum (Extreme Summarization)**

Speculative Decoding เป็นเทคนิคเร่งความเร็วในการสร้าง token ของ Language Model โดยให้ "draft model" ขนาดเล็กสร้าง token หลาย ๆ ตัวล่วงหน้า แล้วให้ "verifier" ตรวจสอบทีเดียว แทนที่จะสร้างทีละ token

ในงานนี้ใช้ **N-gram Drafter** แทน draft model และใช้ **Greedy Verifier** ตรวจสอบกับ ground truth โดยตรง

---

## 2. Dataset: XSum

| รายการ | รายละเอียด |
|--------|------------|
| ชื่อ | Extreme Summarization (XSum) |
| ประเภท | Abstractive Summarization |
| แหล่งข้อมูล | BBC News articles |
| ลักษณะ | summary คือ **หนึ่งประโยคที่เขียนใหม่** โดย editor ไม่ได้ copy จาก article |
| ตัวอย่าง | article: "Flooding has caused widespread damage across the Scottish Borders..." → summary: "Clean-up operations are continuing across the Scottish Borders after flooding caused by Storm Frank." |

ความแตกต่างหลักจาก SQuAD คือ summary ถูก paraphrase ใหม่ ทำให้คำและโครงสร้างประโยคต่างจาก article

---

## 3. การกำหนด Corpus และ Target

| บทบาท | ข้อมูลที่ใช้ | เหตุผล |
|--------|-------------|--------|
| **Corpus** | document (บทความข่าว BBC) | N-gram Drafter ค้นหา pattern ใน article |
| **Target** | summary (ประโยคสรุป) | Verifier ตรวจสอบว่า draft ตรงกับ summary หรือไม่ |

ปัญหาหลักคือ summary ถูกเขียนใหม่โดย editor ดังนั้น tokens ใน summary จำนวนมากไม่ปรากฏใน article เลย

---

## 4. กลไกการทำงาน: N-gram Speculative Decoding

### 4.1 ขั้นตอนเหมือน SQuAD แต่ผลต่างกัน

```
Input:
  corpus_tokens = tokenize(article)    # ~482 tokens
  target_tokens = tokenize(summary)    # ~24 tokens
  current_prefix = [target_tokens[0]]  # เริ่มจาก token แรก = "Clean"

ทุก step:
  1. Drafter รับ current_prefix
  2. ค้นหา n tokens ท้ายสุดของ prefix ใน corpus
  3. ถ้าเจอ: copy tokens ถัดไป K ตัวเป็น draft
     ถ้าไม่เจอ: ลด n ลง (n-gram backoff) จนถึง n=1
  4. Verifier เทียบ draft กับ target ทีละตัว
  5. เพิ่ม recovery token เข้า prefix
```

### 4.2 ตัวอย่าง Step จริงจาก Sample #0

```
Step 1: prefix = ["Clean"]
  Drafter ค้นหา "Clean" ใน article 482 tokens
  → ไม่เจอเลย! "Clean" ไม่ปรากฏในข่าวชิ้นนี้
  → no_draft → บวก recovery = "-up"

Step 2: prefix = ["Clean", "-up"]
  Drafter ค้นหา ["-up"] → ไม่เจอ → no_draft

Step 3-4: "operations", "are" → ไม่เจอเช่นกัน

Step 5: prefix = [..., "across"]
  Drafter เจอ "across" ใน article
  → copy ต่อ: ["the", "Scottish", "Borders"] ← ถูกต้องทั้งสามคำ ✓

Step 6: prefix = [..., "Borders"]
  Drafter เจอ "Borders" ใน article
  → copy ต่อ: ["region", "on", "Tuesday"]  ← มาจาก article
  แต่ summary ต้องการ: ["and", "Dumfries", "and"]
  → full_reject ทันที ✗
```

---

## 5. ผลการทดลอง

### 5.1 ค่าชี้วัดหลัก (Sample #0)

| Metric | ค่า |
|--------|-----|
| Lexical Overlap (article ↔ summary) | 66.7% |
| Token Novelty Rate | 33.3% |
| Speedup Ratio (K=3, n=2) | 1.35x |
| Step type ที่พบบ่อย | `no_draft` (47%), `full_reject` (35%) |
| Zero-draft rate | 47% |

### 5.2 ผลจาก 204,011 Samples (Full Dataset)

| Metric | ค่า |
|--------|-----|
| Mean Speedup | 1.14x |
| Std Speedup | 0.12 |
| Min / Max Speedup | 1.00x / 3.25x |
| Mean Token Novelty | 37.2% (std 15.5%) |
| Mean Zero-draft rate | 42.7% (std 16.0%) |

Speedup ต่ำอย่างสม่ำเสมอในทุก sample (`std=0.12`) สะท้อนว่าปัญหาเป็นเชิงโครงสร้างของ task ไม่ใช่ความบังเอิญ

---

## 6. Root Cause Analysis

### 6.1 Root Cause ที่ 1: Token Novelty (~40% ของ Summary Tokens)

Summary ของ XSum เขียนขึ้นใหม่โดย editor ไม่ใช่การคัดลอกจาก article
ส่งผลให้มีจำนวน token มากที่ไม่ปรากฏในข่าวเลย:

```
Article  : "Flooding has caused widespread damage..."
Summary  : "Clean-up operations are continuing..."

Token "Clean"      → ไม่มีในข่าว
Token "operations" → ไม่มีในข่าว
Token "continuing" → ไม่มีในข่าว
```

เมื่อ token แรกของ summary ค้นหาไม่เจอ drafter ไม่มี draft → ต้องใช้ recovery token ทีละตัว → ประสิทธิภาพเทียบเท่า baseline

**Token Novelty Rate** คือ % ของ summary tokens ที่ไม่ปรากฏในข่าว — XSum มีค่านี้สูงกว่า SQuAD มาก

### 6.2 Root Cause ที่ 2: Paraphrase Effect

แม้ token บางตัวจะปรากฏทั้งใน article และ summary (เช่น "Scottish", "Borders", "flooding") แต่ **context ถัดไปต่างกัน**:

| ตำแหน่งใน Article | ตำแหน่งใน Summary |
|-------------------|-------------------|
| "Scottish Borders **region** on Tuesday" | "Scottish Borders **and** Dumfries and Galloway" |
| "flooding **has** caused widespread" | "flooding **caused** by Storm Frank" |

Drafter copy context จาก article แต่ summary ใช้ paraphrase → full_reject ทันที

### 6.3 การจำแนกประเภท Mismatch

| ประเภท | ความหมาย | สัดส่วนโดยประมาณ |
|--------|----------|-----------------|
| `novel word` | token ที่ผิดไม่มีใน article เลย | ~40% |
| `paraphrase` | token ที่ผิดมีใน article แต่ summary เลือกคำต่างออกไป | ~60% |

ทั้งสองประเภทเกิดจาก abstractive nature ของ task ไม่ใช่ปัญหาเชิงเทคนิค

### 6.4 ผลของ N-gram Size (n)

| n | พฤติกรรม | Speedup |
|---|----------|---------|
| n=1 | หาเจอบ่อย แต่ draft ผิดเกือบทั้งหมด | ต่ำมาก |
| n=2 | balance ที่ดีที่สุดสำหรับ XSum | ปานกลาง ← optimal |
| n=3 | pattern ยาวเกินไป ค้นหาไม่เจอบ่อย | ต่ำ |
| n=4 | ค้นหาไม่เจอแทบทั้งหมด | ต่ำมาก |

XSum ใช้ n=2 เป็น optimal เพราะ summary มี token น้อยที่ตรงกับ article ทำให้ n ขนาดใหญ่ยิ่งหาไม่เจอ

### 6.5 Corpus Size Sensitivity

การทดสอบพบว่าการเพิ่ม corpus จาก 10% เป็น 100% ของ article **ไม่ช่วยเพิ่ม speedup อย่างมีนัยสำคัญ**

สรุปว่าปัญหาหลักไม่ใช่ corpus เล็กเกินไป แต่เป็น **abstractive nature ของ task** ที่ทำให้ pattern ใน summary ไม่ตรงกับ pattern ใน article

---

## 7. เปรียบเทียบกับ SQuAD

| มิติ | SQuAD | XSum |
|------|-------|------|
| Task type | Extractive QA | Abstractive Summarization |
| Lexical overlap | ~100% | ~40–60% |
| Token novelty rate | ~0% | ~30–50% |
| Speedup | ~2x | ~1.1x |
| Step type หลัก | full_accept, partial | no_draft, full_reject |
| ปัญหาหลัก | Boundary overshoot | Token novelty + Paraphrase |
| ข้อสรุป | เหมาะมาก | ไม่เหมาะ |

---

## 8. สรุป

XSum แสดงให้เห็นข้อจำกัดพื้นฐานของ N-gram Speculative Decoding: **drafter ออกแบบมาสำหรับ exact match แต่ abstractive task ใช้ paraphrase**

ผลที่ได้ (speedup ~1.1x) สะท้อนว่า n-gram drafter แทบไม่ได้ประโยชน์จาก corpus ที่มีอยู่ เพราะ pattern ของ summary ไม่ซ้ำกับ article

| จุดที่เรียนรู้ | ผลที่ได้ |
|---------------|---------|
| Token novelty สูง | no_draft rate สูง → speedup ต่ำ |
| Paraphrase effect | full_reject บ่อย แม้เจอ token |
| Corpus ใหญ่ขึ้นไม่ช่วย | ยืนยันว่าปัญหาคือ task type ไม่ใช่ data size |

> **ข้อสรุป**: N-gram Drafter ไม่เหมาะกับ abstractive task หากต้องการ speedup จริงต้องเปลี่ยนไปใช้ small neural LM เป็น drafter แทน เพื่อให้เข้าใจ paraphrase และ semantic similarity ได้
