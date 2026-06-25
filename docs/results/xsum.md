================================================================
เอกสารประกอบการนำเสนอ
Speculative Decoding — XSum Dataset
Root Cause Analysis (Full Dataset: 204,011 samples)
================================================================

ไฟล์กราฟ:
  experiments/xsum/artifacts/xsum_part1.png
  experiments/xsum/artifacts/xsum_part2.png
ข้อมูลดิบ:
  experiments/xsum/full_analysis.json


================================================================
PART 1 — กราฟที่ 1: Step Type Breakdown per K
(xsum_part1.png — ซ้าย)
================================================================

กราฟนี้คืออะไร:
  Stacked bar chart แสดงสัดส่วนของ step type ทั้ง 4 ประเภท
  เมื่อเปลี่ยน draft size K=1,2,3,4
  แต่ละ bar รวมกันได้ 100% คือ step ทั้งหมดใน dataset

Step types ที่เห็น:
  No Draft    (เทา)  — drafter หา n-gram match ไม่เจอเลย ไม่มี draft ออกมา
  Full Reject (แดง)  — มี draft แต่ verifier reject ทุกตัว
  Partial     (ส้ม)  — accept บางตัว
  Full Accept (เขียว) — accept ทุกตัว K tokens ใน 1 step

สิ่งที่เห็นในกราฟ:
  - No Draft  ≈ 40% คงที่ทุก K ไม่ลดลงเลยแม้ K จะเพิ่ม
  - Full Reject ≈ 50% ครองพื้นที่ส่วนใหญ่
  - Full Accept ≈ 2-5% เท่านั้น และลดลงเมื่อ K ใหญ่ขึ้น
  - Partial ≈ 5-8% เพิ่มขึ้นเล็กน้อยตาม K

การตีความ:
  เกือบ 90% ของ step ทั้งหมดได้ประโยชน์เป็น 0 (no_draft + full_reject)
  การเพิ่ม K ไม่ช่วยอะไรเพราะ no_draft ไม่ลดลง
  ปัญหาไม่ใช่ K เล็กเกินไป แต่เป็นว่า drafter หา match ไม่เจอ
  หรือ match ได้แต่คำที่ propose ผิด

Root cause จากกราฟนี้:
  No Draft สูง → summary เริ่มต้นด้วยคำที่ไม่มีในบทความ
  Full Reject สูง → แม้ดราฟต์ได้ แต่ summary ใช้คำ paraphrase ต่างออกไป


================================================================
PART 1 — กราฟที่ 2: Accepted Tokens per Step (K=3)
(xsum_part1.png — กลาง)
================================================================

กราฟนี้คืออะไร:
  Histogram แสดงว่าในแต่ละ speculative step
  ระบบ accept ได้กี่ token (0, 1, 2, หรือ 3)
  รวมทุก step จาก 204,011 samples

สิ่งที่เห็นในกราฟ:
  accepted=0 : ~90% ของ step ทั้งหมด  ← bar สูงมาก
  accepted=1 : ~6%
  accepted=2 : ~3%
  accepted=3 : ~1%

การตีความ:
  9 ใน 10 step ได้ 0 token จาก draft = เท่ากับ baseline ทุกประการ
  มีเพียง 10% เท่านั้นที่ speculative decoding ให้ประโยชน์จริง
  ที่ accepted=0 สูงมาก เพราะ no_draft + full_reject รวมกัน ≈ 90%

Root cause จากกราฟนี้:
  ระบบ speculative decoding ทำงานเป็น baseline แทบทั้งหมด
  ประโยชน์ที่ได้จาก 10% ที่เหลือ บวกกันได้ speedup เฉลี่ยแค่ 1.14x


================================================================
PART 1 — กราฟที่ 3: Mean Speedup vs N-gram Size n (K=3)
(xsum_part1.png — ขวา)
================================================================

กราฟนี้คืออะไร:
  เปลี่ยนขนาด n-gram (n=1,2,3,4) แล้วดู mean speedup
  ที่ K=3 คงที่ ค่าเป็น mean จาก 204,011 samples

สิ่งที่เห็นในกราฟ:
  n=1 : 1.00x  (ไม่ได้ประโยชน์เลย)
  n=2 : 1.14x  (กระโดดขึ้น)
  n=3 : 1.15x  (เพิ่มขึ้นเล็กน้อย)
  n=4 : 1.15x  (หยุดนิ่ง — plateau)

การตีความ:
  n=1 (unigram) ไม่ทำงานเลย — NGramDrafter ไม่ generate draft
  เมื่อเปลี่ยนจาก n=2 เป็น n=3,4 speedup แทบไม่เปลี่ยน
  แสดงว่าการเพิ่ม n ไม่ช่วยให้ match ได้ดีขึ้น

Root cause จากกราฟนี้:
  ปัญหาไม่ใช่ context ไม่พอ (n เล็กเกินไป)
  ปัญหาคือ vocabulary ที่ต่างกัน — ไม่ว่า n จะเท่าไร
  summary ใช้คำที่ drafter ไม่มีใน article อยู่ดี


================================================================
PART 2 — กราฟที่ 4: Speedup Across 204,011 Samples
(xsum_part2.png — ซ้าย)
================================================================

กราฟนี้คืออะไร:
  Bar chart แสดง speedup ของทุก sample เรียงจากมากไปน้อย
  เส้นประสีน้ำเงิน = mean speedup

สิ่งที่เห็นในกราฟ:
  Mean = 1.14x (เส้นประสีน้ำเงิน)
  Baseline = 1.0x (เส้นประสีเทา)
  ส่วนใหญ่ติดอยู่แถว 1.0x-1.2x เป็นกลุ่มหนาแน่น
  มี outlier บางตัวสูงถึง ~3.25x แต่หายากมาก
  การกระจายแคบมาก (std=0.12) — ทุก sample ให้ผลคล้ายกัน

การตีความ:
  ปัญหาเกิดทั่วทั้ง dataset ไม่ใช่แค่บาง article
  Variance ต่ำแสดงว่านี่คือ structural problem ของ task ไม่ใช่ความบังเอิญ
  แม้ article ที่ดีที่สุดก็ได้ speedup ไม่มาก

Root cause จากกราฟนี้:
  ทุก BBC article มีปัญหาเดียวกัน — journalist เสมอเขียน
  summary ด้วยคำใหม่ ทำให้ n-gram drafter แทบช่วยไม่ได้


================================================================
PART 2 — กราฟที่ 5: Speedup vs Token Novelty Rate
(xsum_part2.png — กลาง)
================================================================

กราฟนี้คืออะไร:
  Scatter plot ระหว่าง token novelty (x-axis) กับ speedup (y-axis)
  แต่ละจุด = 1 sample (สุ่ม 500 จุดจาก 204,011)
  token novelty = % ของ summary tokens ที่ไม่มีในบทความเลย

สิ่งที่เห็นในกราฟ:
  จุดกระจายกว้างตลอดช่วง novelty 0-80%
  Speedup ส่วนใหญ่อยู่แถว 1.0x-1.4x ไม่ว่า novelty จะเป็นเท่าไร
  ไม่มี trend ชัดเจน — correlation อ่อนมาก

การตีความ:
  แม้ article ที่มี novelty ต่ำ (summary คล้าย article) ก็ยังได้ speedup ต่ำ
  เพราะแม้ token จะมีใน article แต่ context ลำดับที่ถัดไปต่างกัน (paraphrase)
  Token novelty ไม่ใช่ตัวทำนาย speedup ที่ดีพอ เพราะ
  paraphrase effect สำคัญไม่แพ้ novelty

Root cause จากกราฟนี้:
  มี 2 ปัญหาซ้อนกัน:
  1. Token novelty — คำใหม่ที่ไม่มีในบทความ
  2. Paraphrase — คำมีในบทความ แต่ context ถัดไปต่างกัน
  กราฟนี้แสดงว่าแม้จะแก้ปัญหา novelty ได้ paraphrase ก็ยังทำให้ speedup ต่ำ


================================================================
PART 2 — กราฟที่ 6: Corpus Size Sensitivity
(xsum_part2.png — ขวา)
================================================================

กราฟนี้คืออะไร:
  ทดสอบว่าถ้าให้ drafter เห็นแค่ X% ของบทความ speedup เปลี่ยนไหม
  ค่าเป็น mean ข้าม 204,011 samples ที่แต่ละ fraction

สิ่งที่เห็นในกราฟ:
  10%  corpus : 1.02x
  25%  corpus : 1.07x
  50%  corpus : 1.10x
  75%  corpus : 1.12x
  100% corpus : 1.14x

การตีความ:
  Speedup เพิ่มขึ้นช้ามากเมื่อเพิ่ม corpus
  แม้ใช้บทความ 100% ก็ได้แค่ 1.14x
  เส้นโค้งเพิ่มขึ้นแบบ diminishing return ชันตอนต้น แต่แบนเมื่อใกล้ 100%

Root cause จากกราฟนี้:
  ปัญหาไม่ใช่ corpus ไม่พอ แต่เป็น task type
  ไม่ว่าจะให้ drafter เห็นบทความมากแค่ไหน
  summary ยังคงใช้คำ paraphrase ที่ไม่ match กับบทความ
  เปรียบเทียบกับ SQuAD: corpus sensitivity สูงมาก → คนละ root cause


================================================================
สรุป ROOT CAUSE ANALYSIS — XSum
================================================================

Root Cause หลักที่ 1: Abstractive Nature of Task
  XSum summary ถูกเขียนขึ้นใหม่โดย BBC journalist
  ไม่ใช่การ copy หรือ paraphrase เล็กน้อย แต่เป็นการเขียนใหม่ทั้งหมด
  ผล: token novelty เฉลี่ย 37.2% — เกือบ 4 ใน 10 คำไม่มีในบทความ

Root Cause หลักที่ 2: Paraphrase Effect
  แม้คำจะมีในบทความ แต่ context ถัดไปต่างกัน
  เช่น บทความ: "... flooding has caused widespread damage ..."
       summary: "... flooding caused by Storm Frank ..."
  token "flooding" มีทั้งสองที่ แต่คำต่อไปต่างกัน → full_reject

Root Cause หลักที่ 3: First-Token Failure
  XSum summary มักเริ่มด้วยคำใหม่ที่ไม่มีในบทความ
  เช่น "Clean-up operations are continuing..."
  → "Clean" ไม่มีในบทความ → step แรกเป็น no_draft เสมอ
  → ทำให้ prefix ที่ใช้สำหรับ draft step ถัดไปก็เสียหายไปด้วย

สรุปตัวเลขสำคัญ (full dataset, 204,011 samples, K=3, n=2):

  Lexical Overlap      : mean=62.8%  std=15.5%
    → token ใน summary ที่ปรากฏอยู่ที่ไหนสักที่ใน article
    → ค่าสูง = drafter มีโอกาสหา match ได้ / ค่าต่ำ = ทำงานแทบไม่ได้

  Token Novelty Rate   : mean=37.2%  std=15.5%
    → token ใน summary ที่ไม่มีในบทความเลย (= 1 - Lexical Overlap)
    → root cause หลักที่ทำให้ no_draft สูง

  Speedup Ratio (K=3, n=2) : mean=1.14x  std=0.12
    → ระบบเร็วขึ้นแค่ 14% เทียบกับ baseline

  Acceptance Rate      : 4.5% per draft token
    → จาก draft token ทั้งหมดที่ propose มีแค่ 4.5% ที่ถูก accept
    → mean accepted per step = 0.134 tokens (จาก K=3 ที่เป็นไปได้)

  Zero-draft Rate      : mean=42.7%  std=16.0%
    → 42.7% ของ step ทั้งหมด drafter หา n-gram match ไม่เจอเลย

  Step Types (K=3):
    no_draft    : 42.7%  — ไม่มี draft ออกมา
    full_reject : 47.5%  — draft ถูก reject ทั้งหมด
    partial     :  8.6%  — accept บางส่วน
    full_accept :  1.1%  — accept ครบทุกตัว

  N-gram Speedup (K=3):
    n=1 : 1.00x  (drafter ไม่ทำงานกับ n=1)
    n=2 : 1.14x  ← optimal
    n=3 : 1.15x
    n=4 : 1.15x  (plateau — n ใหญ่ขึ้นไม่ช่วย)

  N-gram Usage:
    เกือบทุก match ที่เกิดขึ้นใช้ n=1 (backoff สูงสุด)
    แสดงว่า bigram/trigram context ของ summary ไม่ตรงกับที่มีใน article

ข้อสรุป:
  N-gram Speculative Decoding ไม่เหมาะกับ Abstractive Summarization
  เพราะ drafter ต้องการ exact token match จาก corpus
  แต่ abstractive task สร้าง output ที่ semantic เหมือนกัน แต่ surface form ต่างกัน
  ถ้าต้องการ speedup จริง ควรเปลี่ยนไปใช้ small neural LM เป็น drafter
  เพื่อให้สามารถ predict paraphrase ได้
