# เอกสารประกอบการนำเสนอ
## Speculative Decoding — กรณีศึกษา: SQuAD Dataset

---

## 1. ภาพรวม

งานนี้ศึกษาการจำลอง **Speculative Decoding** โดยใช้ N-gram Drafter กับ dataset ประเภท Extractive Question Answering ชื่อว่า **SQuAD (Stanford Question Answering Dataset)**

Speculative Decoding เป็นเทคนิคเร่งความเร็วในการสร้าง token ของ Language Model โดยให้ "draft model" ขนาดเล็กสร้าง token หลาย ๆ ตัวล่วงหน้า แล้วให้ "verifier" ตรวจสอบทีเดียว แทนที่จะสร้างทีละ token

ในงานนี้ใช้ **N-gram Drafter** แทน draft model และใช้ **Greedy Verifier** ตรวจสอบกับ ground truth โดยตรง

---

## 2. Dataset: SQuAD

| รายการ | รายละเอียด |
|--------|------------|
| ชื่อ | Stanford Question Answering Dataset (SQuAD) |
| ประเภท | Extractive Question Answering |
| แหล่งข้อมูล | Wikipedia passages |
| ลักษณะ | คำตอบคือข้อความที่ **ดึงมาจาก passage ตรง ๆ** ไม่ได้เขียนใหม่ |
| ตัวอย่าง | passage: "…John of Chelles began construction in 1265…" → question: "Who began construction?" → answer: "John of Chelles" |

---

## 3. การกำหนด Corpus และ Target

| บทบาท | ข้อมูลที่ใช้ | เหตุผล |
|--------|-------------|--------|
| **Corpus** | passage (บทอ่าน) | N-gram Drafter ค้นหา pattern ใน corpus เพื่อสร้าง draft |
| **Target** | answer (คำตอบ) | Verifier ตรวจสอบว่า draft ตรงกับ answer หรือไม่ |

ไม่ใช้ question เป็น corpus เนื่องจาก question มีความยาวสั้นมาก (~7 tokens) ไม่เพียงพอสำหรับการสร้าง n-gram pattern ที่มีประโยชน์

---

## 4. กลไกการทำงาน: N-gram Speculative Decoding

### 4.1 ขั้นตอนการสร้าง Draft

```
Input:
  corpus_tokens = tokenize(passage)   # ~150 tokens
  target_tokens = tokenize(answer)    # ~3–10 tokens
  current_prefix = [target_tokens[0]] # เริ่มจาก token แรก

ทุก step:
  1. Drafter รับ current_prefix
  2. นำ n tokens ท้ายสุดของ prefix ไปค้นหาใน corpus
  3. ถ้าเจอ: copy tokens ถัดไป K ตัวเป็น draft
     ถ้าไม่เจอ: ลด n ลงทีละ 1 (n-gram backoff) จนถึง n=1
  4. Verifier เทียบ draft กับ target ทีละตัว
     accept ไปเรื่อย ๆ จนกว่าจะผิด
  5. เพิ่ม recovery token (token ถัดไปที่ถูกต้อง) เข้า prefix
```

### 4.2 ประเภทของ Step

| ประเภท | ความหมาย | ผลต่อ speedup |
|--------|----------|---------------|
| `no_draft` | drafter ค้นหาไม่เจอ ไม่มี draft | ไม่ช่วย |
| `full_reject` | draft มี แต่ token แรกผิดทันที | ไม่ช่วย |
| `partial` | accept บางส่วน แล้วมิสแมตช์ | ช่วยได้บ้าง |
| `full_accept` | accept draft ทั้งหมด K tokens | ช่วยได้มาก |

---

## 5. ผลการทดลอง

### 5.1 ค่าชี้วัดหลัก (Sample #0)

| Metric | ค่า |
|--------|-----|
| Lexical Overlap (passage ↔ answer) | ~100% |
| Speedup Ratio (K=3, n=3) | ~1.5x – 2.5x |
| Acceptance Rate | ~60–80% |
| Step type ที่พบบ่อย | `partial`, `full_accept` |
| `full_reject` rate | ≈ 0% |

### 5.2 ผลจาก 10 Samples

| Metric | ค่า |
|--------|-----|
| Mean Speedup | ~1.8x – 2.2x |
| Std Dev | ~0.4x |
| Zero-draft rate (เฉลี่ย) | ~10–20% |

Speedup มี variance ปานกลาง เนื่องจากคำตอบสั้น (1–2 tokens) ได้รับประโยชน์น้อยกว่าคำตอบยาว

---

## 6. Root Cause Analysis

### 6.1 เหตุผลที่ได้ผลดี: Extractive Nature

SQuAD เป็น extractive task — คำตอบเป็น substring ของ passage โดยตรง
ดังนั้น N-gram Drafter ทำหน้าที่เป็น **exact span lookup** ไม่ใช่การเดา
เมื่อ drafter เจอ token ที่ตรงกัน tokens ถัดไปใน passage ก็คือ tokens ของคำตอบพอดี

```
Passage : "…the construction was led by John of Chelles in…"
Answer  : "John of Chelles"

ขณะที่ prefix = ["John"]
drafter เจอ "John" ใน passage → copy ["of", "Chelles"] → accept ทั้งคู่ ✓
```

### 6.2 สาเหตุที่บาง Step ยัง Fail

**สาเหตุที่ 1 — Subword Tokenization Context-dependence**

Tokenizer ของ LLM เป็น subword-based (เช่น BPE) คำเดียวกันอาจ tokenize ต่างกันขึ้นกับ context:

```
"Saint Bernadette" ใน passage     → tokens: [24, 3891, 1043, ...]
"Saint" ที่เป็น token แรกของ answer → tokens: [30002, ...]
```

Token ID ต่างกัน → drafter ค้นหาไม่เจอ → `no_draft` → ต้องใช้ recovery token แทน

**สาเหตุที่ 2 — Boundary Overshoot**

Drafter ไม่รู้ว่าคำตอบสิ้นสุดตรงไหน จึงคัดลอก tokens ที่เกินออกมาด้วย:

```
Answer   : "John of Chelles"  [สิ้นสุดที่นี่]
Draft    : ["John", "of", "Chelles", "in", "1265"]  ← copy เกินมา
Verifier : accept 3 ตัวแรก, reject ตัวที่ 4 ("in") → partial step
```

### 6.3 ผลของ N-gram Size (n)

| n | พฤติกรรม | Speedup |
|---|----------|---------|
| n=1 | หาเจอเยอะ แต่ draft ผิดบ่อย (context ไม่ specific) | ต่ำ |
| n=2 | balance ระหว่าง recall กับ precision | ปานกลาง |
| n=3 | เจาะจงขึ้น draft ถูกต้องมากขึ้น | สูง ← optimal |
| n=4 | pattern ยาวเกินไป ค้นหาไม่เจอบ่อย | ลดลง |

---

## 7. สรุป

SQuAD เป็นกรณีที่ N-gram Speculative Decoding ทำงานได้ดีที่สุด เนื่องจาก **task structure ตรงกับ design ของ drafter** — คำตอบเป็น substring ของ corpus ทำให้ drafter ทำงานเหมือน exact lookup และได้รับ draft ที่ถูกต้องสูง

| จุดแข็ง | จุดจำกัด |
|---------|---------|
| Acceptance rate สูง (~70%) | Subword tokenization mismatch |
| แทบไม่มี full_reject | Boundary overshoot ที่ท้ายคำตอบ |
| Speedup เฉลี่ย ~2x | คำตอบสั้นได้ประโยชน์น้อย |

> **ข้อสรุป**: N-gram Drafter เหมาะกับ extractive task ซึ่ง output เป็น substring ของ input
