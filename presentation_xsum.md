# Script พรีเซนต์ — XSum Dataset
> ผู้พูด: [ชื่อเพื่อน]  |  เวลา: ~5 นาที

---

## ส่วนที่ 1 — Dataset คืออะไร (30 วิ)

> "ผมจะพรีเซนต์ Speculative Decoding กับ **XSum** ครับ
> XSum คือ dataset สรุปข่าว BBC โดย summary เป็น **หนึ่งประโยคที่เขียนใหม่**
> ไม่ได้ copy มาจาก article ตรง ๆ เรียกว่า Abstractive Summarization ครับ"

---

## ส่วนที่ 2 — Setup และกลไก (1.5 นาที)

> "Setup เหมือนกันกับ SQuAD ครับ แต่ต่างกันในแก่นแท้:
> - **Corpus** = article (ข่าว BBC)
> - **Target** = summary (ประโยคสรุป)"

> "ปัญหาคือ summary ถูกเขียนใหม่โดย editor ไม่ใช่ copy จาก article
> ลองดูตัวอย่างจริงครับ:"

```
Article : "Flooding has caused widespread damage across
           the Scottish Borders region on Tuesday..."

Summary : "Clean-up operations are continuing across
           the Scottish Borders after flooding..."
```

> "คำว่า 'Clean' ซึ่งเป็นคำแรกของ summary — **ไม่มีในข่าวเลยครับ**
> drafter ค้นหาไม่เจอ → no_draft ตั้งแต่ step แรก"

> "และแม้เจอคำบางคำเช่น 'Borders' ใน article แต่ article ตามด้วย 'region on Tuesday'
> ขณะที่ summary ต้องการ 'and Dumfries and' — context ต่างกัน → full_reject"

---

## ส่วนที่ 3 — ผลลัพธ์และกราฟ (1.5 นาที)

> "มาดูผลครับ"

- Lexical Overlap: **~40-60%** — ต่ำกว่า SQuAD มาก
- Token Novelty Rate: **~30-50%** ของ summary tokens ไม่มีใน article เลย
- Speedup: **~1.1x** — แทบไม่ต่างจาก baseline
- Step type ที่พบบ่อย: `no_draft` + `full_reject` เกือบทั้งหมด

> "ชี้กราฟ Panel 1: no_draft สูงมาก ต่างจาก SQuAD อย่างชัดเจน
> ชี้กราฟ Panel 3: corpus sensitivity — เพิ่ม corpus เป็น 2 เท่า speedup ก็ไม่ขยับ
> แปลว่าปัญหาไม่ใช่ corpus เล็กเกินไป แต่เป็นธรรมชาติของ task"

---

## ส่วนที่ 4 — Root Cause Analysis (1.5 นาที)

**Root Cause 1: Token Novelty (~40% ของ tokens ไม่มีในข่าว)**
> "summary เขียนใหม่ด้วยคำที่ไม่ได้อยู่ใน article
> drafter หาไม่เจอ → no_draft ตลอด → recovery token ทีละตัว → ช้าเหมือน baseline"

**Root Cause 2: Paraphrase Effect**
> "แม้ token เดียวกันปรากฏใน article แต่ context ถัดไปต่างกัน
> เพราะ editor เลือกคำต่างออกไป → full_reject ทันที
> ทั้งสองสาเหตุมาจาก abstractive nature ของ task ครับ ไม่ใช่ bug"

---

## สรุป (30 วิ)

> "สรุปครับ XSum แสดงให้เห็นว่า n-gram speculative decoding **ไม่เหมาะกับ abstractive task**
> เพราะ drafter ออกแบบมาสำหรับ exact match แต่ abstractive summary ใช้ paraphrase
> ถ้าอยากให้ได้ผล ต้องเปลี่ยน drafter เป็น small neural LM ที่เข้าใจ paraphrase ได้ครับ"

---

## Q&A Cheatsheet

| คำถาม | คำตอบสั้น |
|-------|-----------|
| เพิ่ม corpus ช่วยไหม? | ไม่ครับ ปัญหาคือ summary เขียนใหม่ ไม่ใช่ corpus เล็ก |
| lexical overlap 50% ทำไม speedup แค่ 1.1x? | overlap ต้องมาพร้อม context ถูกต้องด้วย แค่ token ตรงไม่พอ |
| แก้ยังไง? | เปลี่ยน drafter เป็น neural model แทน n-gram |
