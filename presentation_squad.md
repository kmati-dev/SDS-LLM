# Script พรีเซนต์ — SQuAD Dataset
> ผู้พูด: [ชื่อคุณ]  |  เวลา: ~5 นาที

---

## ส่วนที่ 1 — Dataset คืออะไร (30 วิ)

> "ผมจะพรีเซนต์ Speculative Decoding กับ **SQuAD** ครับ
> SQuAD คือ dataset ถาม-ตอบ ที่คำตอบเป็นข้อความที่ดึงมาจาก passage ตรง ๆ
> เรียกว่า Extractive QA ครับ"

---

## ส่วนที่ 2 — Setup และกลไก (1.5 นาที)

> "ใน setup ของเรา:
> - **Corpus** = passage (บทอ่าน) — ใช้สร้าง n-gram
> - **Target** = answer (คำตอบ) — ใช้ตรวจสอบ"

> "กลไกง่าย ๆ คือ drafter ค้นหา n-gram ท้ายสุดของ prefix ใน passage
> ถ้าเจอก็ copy tokens ถัดไป K ตัวเป็น draft
> verifier เทียบกับ answer ทีละตัว — accept จนกว่าจะผิด"

> "เหตุผลที่มันทำงานได้ดีกับ SQuAD คือ **answer คือ substring ของ passage ตรง ๆ**
> ดังนั้น drafter ก็แค่ค้นหาแล้ว copy ไม่ต้องเดาครับ"

---

## ส่วนที่ 3 — ผลลัพธ์และกราฟ (1.5 นาที)

> "มาดูผลครับ"

- Lexical Overlap: **~100%** — token ใน answer มีใน passage เกือบทั้งหมด
- Speedup: **~1.5x – 2.5x** ขึ้นอยู่กับความยาวของคำตอบ
- Step type ที่พบบ่อย: `partial` และ `full_accept` — แทบไม่มี `full_reject` เลย

> "ชี้กราฟ Panel 1: step breakdown เห็นว่า full_reject แทบ 0%
> ชี้กราฟ Panel 2: acceptance histogram เห็นว่าส่วนใหญ่ accept 2-3 tokens ต่อ step"

---

## ส่วนที่ 4 — Root Cause Analysis (1.5 นาที)

**ทำไมถึงได้ผลดี?**
> "เพราะ task เป็น extractive — answer = copy จาก passage
> drafter ทำงานเหมือน exact lookup ไม่ใช่การเดาครับ"

**ทำไมบางครั้งยัง fail?**
> "มีสองสาเหตุหลักครับ:
>
> 1. **Subword tokenization mismatch** — คำเดียวกัน เช่น 'Saint' tokenize ต่างกันเมื่ออยู่ต้นประโยค vs กลางประโยค → ค้นหาไม่เจอ → `no_draft`
>
> 2. **Boundary overshoot** — คำตอบจบแล้ว แต่ drafter ไม่รู้ เลย copy เนื้อหาจาก passage ต่อ → `partial` step"

---

## สรุป (30 วิ)

> "สรุปครับ SQuAD เป็น use case ที่เหมาะกับ n-gram speculative decoding มากที่สุด
> เพราะ task structure ตรงกับ design ของ drafter พอดี
> Speedup ~2x ด้วย n-gram เล็ก ๆ โดยไม่ต้องใช้ neural model เลยครับ"

---

## Q&A Cheatsheet

| คำถาม | คำตอบสั้น |
|-------|-----------|
| ทำไมไม่ใช้ question ด้วย? | question สั้นมาก สร้าง n-gram ได้น้อย passage มีคำตอบอยู่ |
| speedup 2x พอไหม? | นี่คือ n-gram เล็ก ๆ ถ้าใช้ neural drafter จริงจะสูงกว่านี้ |
| ปรับ n ได้ไหม? | ได้ แก้ที่ `experiments/squad/config.json` ค่า `n_gram_size` |
