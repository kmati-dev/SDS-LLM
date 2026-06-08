================================================================
เอกสารประกอบการนำเสนอ
Speculative Decoding — SQuAD Dataset
Root Cause Analysis (Full Dataset: 81,770 samples)
================================================================

ไฟล์กราฟ:
  experiments/squad/artifacts/squad_part1.png
  experiments/squad/artifacts/squad_part2.png
ข้อมูลดิบ:
  experiments/squad/full_analysis.json


================================================================
PART 1 — กราฟที่ 1: Step Type Breakdown per K
(squad_part1.png — ซ้าย)
================================================================

กราฟนี้คืออะไร:
  Stacked bar chart แสดงสัดส่วนของ step type ทั้ง 4 ประเภท
  เมื่อเปลี่ยน draft size K=1,2,3,4,5,6
  ค่าเป็น mean จาก 81,770 samples

Step types ที่เห็น:
  No Draft    (เทา)  — drafter หา n-gram match ไม่เจอ
  Full Reject (แดง)  — draft ถูก reject ทั้งหมด
  Partial     (ส้ม)  — accept บางตัว
  Full Accept (เขียว) — accept ทุกตัว K tokens ใน 1 step

สิ่งที่เห็นในกราฟ:
  K=1: Full Accept ≈ 49% (สูงมาก), No Draft ≈ 43%, Full Reject ≈ 8%
  K=2: Full Accept ลดลงเหลือ 33%, Partial เพิ่มขึ้น
  K=3: Full Accept ≈ 22%, Partial ≈ 24%
  K=4-6: Full Accept ลดลงเรื่อย ๆ เหลือ ~8% ที่ K=6
         Partial เพิ่มขึ้นเรื่อย ๆ เป็น ~37% ที่ K=6
  Full Reject คงที่ ~8% ทุก K — ต่ำมาก

การตีความ:
  Full Reject ต่ำมาก (~8%) ต่างจาก XSum (~50%) อย่างชัดเจน
  → เมื่อ drafter find match ได้ token ที่ propose มักถูกต้อง
  No Draft สูง (~43-47%) เพราะ context บางส่วนหาไม่เจอใน passage
  ที่ K เพิ่มขึ้น Full Accept ลดลงเพราะต้องการ K consecutive correct tokens

Root cause จากกราฟนี้:
  SQuAD ทำงานดีเพราะ answer IS ใน passage (extractive)
  ปัญหาหลักคือ No Draft — drafter หา context ไม่เจอ
  ไม่ใช่ Full Reject — เมื่อ draft ได้ มักถูกต้อง


================================================================
PART 1 — กราฟที่ 2: Accepted Tokens per Step (K=3)
(squad_part1.png — กลาง)
================================================================

กราฟนี้คืออะไร:
  Histogram แสดงว่าในแต่ละ step accept ได้กี่ token (0,1,2,3)
  pooled จากทุก step ใน 81,770 samples

สิ่งที่เห็นในกราฟ:
  accepted=0 : 51% — ไม่ได้ประโยชน์ (no_draft + full_reject)
  accepted=1 : 13%
  accepted=2 : 9%
  accepted=3 : 27% — full accept (สูงมาก เทียบกับ XSum ที่ ~1%)

การตีความ:
  Distribution เป็น bimodal — สูงที่ 0 และ 3
  ≈ 1 ใน 4 step ได้ K=3 tokens ทั้งหมด ← ดีมาก
  เมื่อ drafter find match ได้ มักจะ accept ครบทั้ง draft
  ต่างจาก XSum ที่ histogram stack ที่ 0 เกือบทั้งหมด

Root cause จากกราฟนี้:
  ปัญหาหลักคือ 51% ที่ accepted=0 ซึ่งมาจาก no_draft เป็นหลัก
  เมื่อแก้ no_draft ได้ speedup จะสูงขึ้นมาก


================================================================
PART 1 — กราฟที่ 3: Mean Speedup vs N-gram Size n (K=3)
(squad_part1.png — ขวา)
================================================================

กราฟนี้คืออะไร:
  เปลี่ยนขนาด n-gram (n=1,2,3,4) ดู mean speedup ที่ K=3
  ค่าเป็น mean จาก 81,770 samples

สิ่งที่เห็นในกราฟ:
  n=1 : 1.00x  (ไม่ทำงาน)
  n=2 : 1.66x  (กระโดดขึ้นมาก)
  n=3 : 1.72x  (ยังเพิ่มอยู่)
  n=4 : 1.73x  (เพิ่มขึ้นเล็กน้อย ใกล้ plateau)

การตีความ:
  ต่างจาก XSum ที่ plateau ที่ n=2 อย่างชัดเจน
  SQuAD ยังได้ประโยชน์จาก n ที่ใหญ่ขึ้น (n=3 ดีกว่า n=2)
  เพราะ longer n-gram → more precise span identification
  n=3 คือ FIXED_N ที่เลือกใช้ ซึ่ง optimal สำหรับ SQuAD

Root cause จากกราฟนี้:
  SQuAD ตอบสนองต่อ n-gram size เพราะ answer อยู่ใน passage
  ยิ่ง n ใหญ่ → match ที่แม่นยำขึ้น → ลด boundary overshoot
  ต่างจาก XSum ที่ n ไม่ช่วยเพราะ vocabulary ต่างกันโดยพื้นฐาน


================================================================
PART 2 — กราฟที่ 4: Speedup Across 81,770 Samples
(squad_part2.png — ซ้าย)
================================================================

กราฟนี้คืออะไร:
  Bar chart แสดง speedup ของทุก sample เรียงจากมากไปน้อย
  เส้นประสีแดง = mean speedup

สิ่งที่เห็นในกราฟ:
  Mean = 1.72x (เส้นประสีแดง)
  Range กว้างมาก — ต่ำสุด 1.0x สูงสุด ~4.0x
  การกระจายค่อนข้าง gradual ไม่ cluster ที่จุดเดียว
  มี samples ที่ได้ speedup สูงจำนวนมาก (เส้นโค้งลงช้า)

การตีความ:
  Variance สูง (std ≈ 0.5x) — ต่างจาก XSum ที่ std=0.12x
  บาง sample ได้ >3x เพราะ answer ยาวและอยู่ต้น passage
  บาง sample ได้ ~1x เพราะ answer สั้นมาก (1-2 tokens) หรืออยู่ท้าย passage
  ความหลากหลายนี้สะท้อนความแตกต่างของ answer length ใน SQuAD

Root cause จากกราฟนี้:
  Speedup ขึ้นอยู่กับ answer length และตำแหน่งใน passage
  Short answer → น้อย step → speedup ต่ำ
  Long answer → หลาย step → speedup สูง


================================================================
PART 2 — กราฟที่ 5: Speedup vs Lexical Overlap
(squad_part2.png — กลาง)
================================================================

กราฟนี้คืออะไร:
  Scatter plot ระหว่าง lexical overlap (x-axis) กับ speedup (y-axis)
  500 จุดสุ่มจาก 81,770 samples
  lexical overlap = % ของ answer tokens ที่ปรากฏในบางที่ใน passage

สิ่งที่เห็นในกราฟ:
  Positive trend — overlap สูง → speedup สูงกว่า
  Cluster หนาแน่นที่ overlap สูง (80-100%) — SQuAD ส่วนใหญ่ overlap สูง
  บางจุดที่ overlap 100% ได้ speedup สูงถึง 3-4x
  จุดที่ overlap ต่ำ (0-40%) speedup มักอยู่แถว 1.0-1.5x

การตีความ:
  Overlap สูง → tokens ของ answer มีใน passage → drafter หา match ได้
  Positive correlation ชัดเจนกว่า XSum ที่ scatter ไม่มี trend
  Overlap ≈ 100% ไม่ได้การันตี speedup สูงเสมอไป เพราะยังมี no_draft จาก
  context mismatch (boundary overshoot effect)

Root cause จากกราฟนี้:
  Lexical overlap เป็น predictor ที่ดีของ speedup ใน SQuAD
  ตรงข้ามกับ XSum ที่ overlap ไม่ช่วยเพราะ paraphrase ซ้อนทับ


================================================================
PART 2 — กราฟที่ 6: Corpus Size Sensitivity
(squad_part2.png — ขวา)
================================================================

กราฟนี้คืออะไร:
  ทดสอบว่าถ้าให้ drafter เห็นแค่ X% ของ passage speedup เปลี่ยนแค่ไหน
  ค่าเป็น mean ข้าม 81,770 samples ที่แต่ละ fraction

สิ่งที่เห็นในกราฟ:
  10%  corpus : 1.14x
  25%  corpus : 1.30x
  50%  corpus : 1.47x
  75%  corpus : 1.59x
  100% corpus : 1.72x

การตีความ:
  Strong positive correlation — corpus ใหญ่ขึ้น speedup สูงขึ้นชัดเจน
  จาก 10% → 100% speedup เพิ่มถึง 0.58x (ต่างกัน 51%)
  เส้นโค้งขึ้นแบบ roughly linear ไม่มี saturation ชัดเจน

Root cause จากกราฟนี้:
  Answer อยู่ใน passage → ยิ่ง drafter เห็นมาก → โอกาสหา span ยิ่งสูง
  ถ้า answer อยู่ท้าย passage และ drafter เห็นแค่ 10% → หาไม่เจอ
  นี่คือ contrast กับ XSum ที่ corpus sensitivity ต่ำมาก
  → ยืนยันว่า SQuAD ปัญหาคือ "ไม่เจอ span" ไม่ใช่ "คำต่างกัน"


================================================================
สรุป ROOT CAUSE ANALYSIS — SQuAD
================================================================

Root Cause หลักที่ 1: No Draft จาก Context Mismatch
  SQuAD answer เป็น exact span จาก passage
  แต่ context ที่ drafter ใช้ค้นหาคือ prefix ของ answer (ตอน generate)
  ซึ่งอาจไม่ตรงกับ context รอบ ๆ span ใน passage เสมอไป
  เช่น: answer = "Napoleon" แต่ context prefix = "born in" ในขณะที่
        passage มี "Napoleon was born in Corsica" — tokenization ต่างกัน

Root Cause หลักที่ 2: Boundary Overshoot
  เมื่อ drafter เจอ span ของ answer ใน passage จะ copy ต่อเกิน answer ไป
  เช่น: answer = "1815" แต่ passage มี "in 1815 in France"
        drafter draft = "1815", "in", "France" → accept แค่ "1815"
        แล้ว reject "in", "France" ← boundary overshoot
  ทำให้ K token ที่ draft ไม่ได้ accept ครบ
  มักเกิดกับ short answer ที่อยู่กลาง passage

Root Cause หลักที่ 3: Answer Length Effect
  Short answer (1-2 tokens) → น้อย step → speedup ต่ำ (~1.0-1.2x)
  Long answer (5+ tokens) → หลาย step → drafter มีโอกาสช่วยได้มาก
  ทำให้ variance สูง (std ≈ 0.5x) เพราะ answer length หลากหลาย

สรุปตัวเลขสำคัญ (full dataset, 81,770 samples, K=3, n=3):

  Lexical Overlap      : mean=72.0%  std=27.2%
    → token ใน answer ที่ปรากฏอยู่ที่ไหนสักที่ใน passage
    → สูงกว่า XSum (62.8%) เพราะ answer IS ใน passage

  Speedup Ratio (K=3, n=3) : mean=1.72x  std=0.80
    → เร็วขึ้น 72% เทียบกับ baseline
    → std สูงมากเพราะ answer length หลากหลาย

  Acceptance Rate      : 37.6% per draft token
    → จาก draft token ที่ propose มา 37.6% ถูก accept
    → สูงกว่า XSum (4.5%) มากเพราะ draft ถูกต้องบ่อยกว่า
    → mean accepted per step = 1.129 tokens (จาก K=3 ที่เป็นไปได้)

  Zero-draft Rate      : mean=45.8%  std=35.7%
    → 45.8% ของ step ทั้งหมด drafter หา match ไม่เจอ
    → std สูงมาก — บาง sample เจอได้เยอะ บางอันไม่เจอเลย

  Step Types (K=3):
    no_draft    : 45.8%  — ไม่มี draft ออกมา
    full_reject :  8.1%  — draft ถูก reject ทั้งหมด (ต่ำมาก vs XSum 47.5%)
    partial     : 23.7%  — accept บางส่วน
    full_accept : 22.4%  — accept ครบทุกตัว (สูงมาก vs XSum 1.1%)

  N-gram Speedup (K=3):
    n=1 : 1.00x  (drafter ไม่ทำงานกับ n=1)
    n=2 : 1.66x
    n=3 : 1.72x  ← optimal (FIXED_N)
    n=4 : 1.73x  (ยังเพิ่มอยู่ — ต่างจาก XSum ที่ plateau ที่ n=2)

  N-gram Usage:
    Match ส่วนใหญ่เกิดจาก n=1 backoff เช่นกัน
    แต่เนื่องจาก answer อยู่ใน passage จริง backoff ยังให้ token ที่ถูกได้

เปรียบเทียบกับ XSum:
  SQuAD speedup สูงกว่ามาก (1.72x vs 1.14x)
  เพราะ SQuAD เป็น extractive — answer อยู่ใน corpus
  N-gram drafter ออกแบบมาสำหรับงานแบบนี้โดยตรง
  Corpus sensitivity สูง → ยิ่งมี passage ครบ ยิ่งดี
  Full reject ต่ำ → เมื่อ draft ได้ มักถูกต้อง

ข้อสรุป:
  N-gram Speculative Decoding เหมาะกับ Extractive task อย่าง SQuAD
  ข้อจำกัดหลักที่เหลืออยู่คือ boundary overshoot และ no_draft จาก tokenization
  ซึ่งแก้ได้ด้วย smarter span detection หรือ answer-end marker
