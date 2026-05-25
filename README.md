# Speculate Decoding Simulator (Greedy Speculative Decoding)

---

## 📌 สรุปรายการ Checklist และการจับคู่ไฟล์

| รายการ Checklist | คลาส / ฟังก์ชัน ที่นำไปใช้จริง | ชื่อไฟล์หลัก | รายละเอียดเบื้องต้น |
| :--- | :--- | :--- | :--- |
| **Abstract class ของ Playback** | `AbstractPlayback` | [interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/interfaces.py) | กำหนดอินเทอร์เฟซตัวควบคุมหลัก รองรับ Dependency Injection และการรันแบบทีละโทเค็น |
| **Abstract class ของ Drafter** | `AbstractDrafter` | [interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/interfaces.py) | อินเทอร์เฟซของโมเดลผู้ร่างเพื่อเดาโทเค็นล่วงหน้า |
| **Abstract class ของ Verifier** | `AbstractVerifier` | [interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/interfaces.py) | อินเทอร์เฟซโมเดลผู้ตรวจแบบ Greedy คืนค่า Accepted และ Recovery Token |
| **Implementation n-gram decoding**| `NGramDrafter` | [simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/simulator.py) | คลาสโมเดลผู้ร่างจริงด้วยเทคนิคประวัติ N-gram และมีระบบถอยกลับ (Backoff) |
| **Implementation verifier** | `GreedyVerifier` | [simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/simulator.py) | คลาสผู้ตรวจจริง เปรียบเทียบเรียงตัวจนเจอมิสแมตช์ตัวแรกแล้วดึง Recovery Token |
| **Implementation metrics** | `PlaybackMetrics` | [simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/simulator.py) | คลาสเก็บสถิติคำนวณ ความเร็ว (Speedup), ยอดสูงสุดยอมรับ และค่าเฉลี่ยต่อขั้นตอน |
| **Implementation ของ Playback** | `SpeculativePlayback` | [simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/simulator.py) | ระบบรันจำลองการทับศัพท์/ถอดโทเค็นจริงโดยดึงข้อมูลจาก Tokenizer แบบ Duck Typing |
| **Test case** | pytest unit tests | [test_interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/test_interfaces.py) | รวมเคสทดสอบ 12 เคสครอบคลุมทั้ง Abstract, Concrete, Metrics และ E2E Playback |

---

## 🧠 ลอจิกการทำงานอย่างละเอียด (Detailed Logic)

### 1. Abstract Interfaces (`interfaces.py`)
- **เป้าหมาย**: ป้องกันการสร้างออบเจกต์ (Instantiation) โดยตรง เพื่อใช้ทำหน้าที่เป็นสัญญาร่วม (Interface Contract) ระหว่างส่วนต่าง ๆ ของโปรเจกต์
- **Abstract Constraints**: คลาสแม่ทุกตัวมีเมธอดที่ตกแต่งด้วย `@abstractmethod` ส่งผลให้หากมีการเรียกคำสั่งสร้าง Object จาก Abstract คลาสโดยตรง คอมไพเลอร์ Python จะพ่น `TypeError` ทันที

### 2. N-Gram Drafter (`NGramDrafter` ใน `simulator.py`)
- **เป้าหมาย**: จำลองโมเดลตัวเล็ก (Draft Model) ที่เดาคำถัดไปได้อย่างรวดเร็ว
- **ลอจิกเชิงลึก**:
  1. รับโทเค็นที่ได้สะสมมาล่าสุดในลูปจำลอง (`prompt`).
  2. เริ่มดึงคำต่อท้ายด้วยขนาดที่ต้องการคือ $(N-1)$-gram (เช่น หาก $N=3$ จะดึงโทเค็นย้อนหลังมา 2 ตัวล่าสุด).
  3. ค้นหารูปแบบเดียวกันนี้ในคลังข้อความต้นแบบ (`corpus_tokens`).
  4. หากเจอจุดที่เหมือนกัน มันจะคาดเดาคำถัดไปตามจำนวนร่างที่ตั้งไว้ ($K$ หรือ `draft_size`).
  5. **Backoff Mechanism**: หากไม่พบประวัติตามคู่ $(N-1)$-gram, อัลกอริทึมจะขยับถอยหลังลงมาเรื่อย ๆ ค้นหาเป็น $(N-2)$-gram ไปจนถึง 1-gram (คำเดียวโดด ๆ) เพื่อหาโอกาสในการคาดเดาที่แม่นยำที่สุด หากยังไม่พบเลยจะส่งลิสต์เปล่า `[]` กลับไป

### 3. Greedy Verifier (`GreedyVerifier` ใน `simulator.py`)
- **เป้าหมาย**: เปรียบเทียบโทเค็นที่ Drafter เดาเข้ามาเทียบกับเฉลยจริง (Ground Truth หรือ complete_tokens) เพื่อหาว่าจุดใดที่ผิด
- **ลอจิกเชิงลึก**:
  1. เปรียบเทียบดราฟต์โทเค็น (`draft_tokens`) กับค่าในเฉลยที่ต่อเนื่องจากประโยคปัจจุบัน (`current_prefix`).
  2. ทำการเช็กทีละตำแหน่ง (Index $i$ จาก $0$ ถึง $K-1$) ในลูปแบบ Greedy:
     - หาก $draft\_tokens[i] == complete\_tokens[len(current\_prefix) + i]$: ถือเป็นโทเค็นที่ **ผ่านการยอมรับ (Accept)**.
     - หากตัวแรกใดไม่ตรง (Mismatch): โทเค็นตั้งแต่ตำแหน่งนั้นไปจนสิ้นสุดดราฟต์จะถูก **ปฏิเสธทั้งหมด (Reject)** ทันที และเบรกออกจากลูป.
  3. **Recovery Token**: ค้นหาโทเค็นในจุดที่เกิด mismatch ตัวแรก (หรือโทเค็นตัวถัดไปหลังสุดหลังจากดราฟต์ผ่านทั้งหมด) และแนบเป็น Recovery Token ซึ่งเป็นคำเฉลยกลับคืนไปด้วย เพื่อให้ลูปเดินหน้าต่อได้โดยไม่ติดขัด
  4. ส่งค่าคืนในรูปแบบ Dict:
     ```python
     {
         "accepted_tokens": [ลิสต์โทเค็นที่ดราฟต์ผ่าน] + [Recovery Token],
         "accepted_count": จำนวนที่ยอมรับ,
         "rejected_count": จำนวนที่ปฏิเสธ
     }
     ```

### 4. Metrics Tracker (`PlaybackMetrics` ใน `simulator.py`)
- **เป้าหมาย**: วัดผลทางคณิตศาสตร์ว่าการทำ Speculative Decoding เร็วกว่าปกติจริงหรือไม่
- **ลอจิกเชิงลึก**:
  - `record_step(accepted, rejected)`: เรียกใช้ทุก ๆ ขั้นตอนเพื่ออัปเดตสถิติจำนวนโทเค็นและคำนวณค่าสูงสุดที่ประหยัดได้ต่อ 1 Step (`max_accepted_in_single_step`).
  - **Speedup Ratio**: คำนวณตามสูตร:
    $$\text{Speedup Ratio} = \frac{\text{Normal Steps (จำนวนโทเค็นทั้งหมด)}}{\text{Speculative Steps (ขั้นตอนที่ Verifier Forward จริง)}}$$
    หาก Speedup > 1.0 แสดงว่าประมวลผลเร็วกว่าแบบปกติอย่างมีนัยสำคัญ

### 5. Playback Controller (`SpeculativePlayback` ใน `simulator.py`)
- **เป้าหมาย**: ควบคุมลูปหลักเพื่อจำลองการทับศัพท์ทีละโทเค็นจนกว่าจะได้ข้อความเป้าหมายครบถ้วน
- **ลอจิกเชิงลึก**:
  1. แปลงข้อความเป้าหมายเป็น Token IDs ผ่าน Tokenizer Duck Typing (`self.tokenizer.encode()`).
  2. เริ่มต้นลูปด้วยโทเค็นเริ่มต้นตัวแรกของประโยค.
  3. ภายใต้โหมด **Speculative (`use_drafter=True`)**:
     - ดึงดราฟต์จาก `self.drafter.generate_draft()`.
     - ทำการประเมินใน `self.verifier.verify()`.
     - ขยายประโยคใน prefix ด้วยโทเค็นที่ได้ทั้งหมด (Accepted + Recovery).
  4. ภายใต้โหมด **Normal (`use_drafter=False`)**:
     - เพิ่มโทเค็นเฉลยถัดไปเพียงครั้งละ 1 ตัวตามระบบมาตรฐานทั่วไป.
  5. เมื่อประโยคครบถ้วนแล้ว ทำการแปลงกลับเป็นสตริงข้อความ (`self.tokenizer.decode()`) และส่งกลับออกมา

---

## 📊 ผลลัพธ์และบทวิเคราะห์ทางเทคนิค (Benchmark & Technical Analysis)

เมื่อรันไฟล์เดโมทดสอบ `demo_simulation.py` บนสถาปัตยกรรมตัวจำลองจริง จะพบข้อมูลสถิติที่สำคัญดังนี้:

### 1. แหล่งที่มาของข้อมูลนำเข้า (Inputs & Datasets Source)
ข้อมูลที่เราใส่เข้าไปในตัวจำลองประกอบด้วย 2 ส่วนหลัก:
1. **คลังข้อมูลต้นแบบของ Drafter (`Corpus Tokens: 364`)**:
   - **ที่มา**: เป็นข้อความบทความเชิงเทคนิคภาษาอังกฤษ (Technical WikiText Style) เกี่ยวกับทฤษฎีการทำงานของ *Speculative Decoding*, *Memory-bound Bottleneck* และ *Greedy Verification* ที่ถูกเขียนขึ้นในโค้ด
   - **บทบาท**: ทำหน้าที่จำลอง " logs ข้อความในอดีต" หรือ "คลังเอกสารจัดเก็บเพื่อสืบค้นข้อมูลย้อนหลัง" (Retrieval Database) เพื่อให้ `NGramDrafter` นำไปใช้ดึงแพทเทิร์นการเดาคำถัดไป
2. **ข้อความเป้าหมายรันจำลอง (`Target Tokens: 135`)**:
   - **ที่มา**: ข้อความสั้นที่สกัดโครงสร้างบางส่วนมาจากคลังข้อมูลต้นแบบ เพื่อจำลองสถานการณ์จริงที่ LLM กำลังตอบคำถามหรือสร้างคำใหม่ (Target output sequence)
   - **บทบาท**: ใช้เป็น Ground Truth หรือเฉลยต้นแบบ ซึ่งตัวจำลอง `SpeculativePlayback` จะต้องป้อนข้อความนี้ทีละคำผ่านตัวแปร `complete_tokens` เพื่อเปรียบเทียบกับคำที่เดาอย่างแม่นยำ

---

### 2. ล็อกการทำงานจริงในเทอร์มินัล (Terminal Output Analysis)

```text
Baseline (Normal Decoding): 134 steps, 1.0x Speedup
Speculative (K=1): Steps = 75 | Avg Accept = 0.79 | Speedup = 1.79x
Speculative (K=2): Steps = 64 | Avg Accept = 1.11 | Speedup = 2.09x
Speculative (K=3): Steps = 50 | Avg Accept = 1.68 | Speedup = 2.68x
Speculative (K=4): Steps = 48 | Avg Accept = 1.79 | Speedup = 2.79x
Speculative (K=5): Steps = 43 | Avg Accept = 2.12 | Speedup = 3.12x
```

* **Baseline (1.0x)**: การรันแบบดั้งเดิมของ LLM ซึ่งต้องเรียกใช้โมเดลใหญ่ **134 รอบ** (134 steps) เพื่อสร้างคำใหม่ทีละคำ (1 Token ต่อ 1 Step)
* **K = 1**: ให้ผู้ร่างช่วยเดาคำถัดไป 1 คำล่วงหน้า มีการยอมรับคำที่เดาเฉลี่ยอยู่ที่ 0.79 คำ ส่งผลให้ขั้นตอนรันโมเดลใหญ่ลดเหลือเพียง **75 รอบ** และเร่งความเร็วขึ้น **1.79 เท่า**
* **K = 3 (จุดคุ้มค่าทางเศรษฐศาสตร์)**: ให้เดาคำล่วงหน้าคราวละ 3 คำ ส่งผลให้การยอมรับเฉลี่ยขึ้นไปถึง 1.68 คำ ขั้นตอนการรันโมเดลเดี่ยวลดเหลือเพียง **50 รอบ** และเร่งความเร็วขึ้นอย่างก้าวกระโดดเป็น **2.68 เท่า**
* **K = 5 (ยอดสูงสุดในโหมดจำลอง)**: ให้เดาคำล่วงหน้าคราวละ 5 คำ เมื่อมีประโยคซ้ำซ้อนแมตช์กันอย่างลงตัว ตัวยอมรับขึ้นไปที่ 2.12 คำ ขั้นตอนรันโมเดลลดเหลือต่ำที่สุดที่ **43 รอบ** และได้ความเร็วพุ่งสูงสุดถึง **3.12 เท่า**

---

### 3. บทวิเคราะห์แผนภูมิภาพคู่ (Double-Panel Performance Chart Analysis)

รูปกราฟสถิติที่พล็อตลงบน [speedup_benchmark.png](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/speedup_benchmark.png) แสดงข้อมูล 2 ประเด็นหลัก:

#### ฝั่งซ้าย: Inference Acceleration (Speedup Ratio)
* แสดงตัวคูณอัตราเร่งความเร็วของการประมวลผล (แกน Y) เทียบกับขนาดของกลุ่มคำดราฟต์ $K$ (แกน X)
* **ข้อสังเกตเชิงลึก**: สโลป (Slope) ความชันกราฟพุ่งขึ้นอย่างงดงามในช่วง $K=1 \to K=3$ แต่จะเริ่ม **"แบนราบลงอย่างเห็นได้ชัด" (Flattens Out)** ในช่วงช่วง **$K=3 \to K=4$** (ความเร็วเพิ่มขึ้นเล็กน้อยจาก 2.68x เป็น 2.79x)

#### ฝั่งขวา: Average Speculated Tokens Accepted per Step
* แสดงปริมาณของโทเค็นเฉลี่ยที่ระบบผู้ตรวจยอมรับให้นำมาใช้งานได้จริงในหนึ่งขั้นตอน (แกน Y) เทียบกับขนาด $K$ (แกน X)
* **ข้อสังเกตเชิงลึก**: อัตรายอมรับเฉลี่ยเพิ่มขึ้นน้อยมากในช่วง $K=3 \to K=4$ (เพิ่มขึ้นจาก 1.68 คำ เป็น 1.79 คำ เท่านั้น)

#### 💡 ทำไมจุดคุ้มค่า (Sweet Spot) จึงเป็น $K=3$ หรือ $K=5$?
1. **กฎการลดน้อยถอยลงของผลตอบแทน (Law of Diminishing Returns)**: 
   ในช่วง $K=3 \to K=4$ อัตราความแม่นยำในการเดาแทบไม่เพิ่มขึ้นเลย เนื่องจาก N-gram Drafter หาคำแมตช์ที่มีความยาวระดับ 4 คำได้ยากขึ้น ส่งผลให้โทเค็นที่ 4 ที่ Drafter สร้างขึ้นมามักจะถูก Reject ในลูป Greedy **ดังนั้น $K=3$ จึงเป็นจุดคุ้มค่าสูงสุด (Sweet Spot)** เนื่องจากประหยัดขั้นตอนประมวลผลไปถึง 62% โดยแทบไม่ต้องเสียกำลังในการคำนวณและ Reject คำส่วนเกิน
2. **จุดประหยัดสูงสุด (Max Yield) $K=5$**: 
   ความเร็วพุ่งขึ้นมาอีกครั้งที่ 3.12x เนื่องจากข้อความทดสอบมีบทความยาวช่วงท้ายที่ซ้ำกันกับคลังประวัติ (High Phrase Repetition) ทำให้ Drafter เดาประโยคยาวได้สำเร็จเป็นกรณีพิเศษ

---

## 🚀 วิธีการทดสอบและสั่งใช้งานจริง

### 1. วิธีรันทดสอบ Unit Tests
เปิดเทอร์มินัลเข้าไปที่โฟลเดอร์ของโปรเจกต์และพิมพ์คำสั่ง:
```bash
cd /Users/nantaporn/Documents/indiv-llm/spec-decode-greedy
python3 -m pytest test_interfaces.py -v
```

### 2. วิธีรันเพื่อวาดกราฟ Speedup Ratio จริง
เราได้จัดทำตัวรันเดโมที่โหลดโมเดลจริงและคำนวณสถิติพร้อมออกผลเป็นกราฟสุดพรีเมียม ให้พิมพ์สั่งงานที่เทอร์มินัล:
```bash
python3 demo_simulation.py --tokenizer gpt2 --max_draft 5
```
หลังจากประมวลผลเสร็จสิ้น ระบบจะทำการสร้างและอัปเดตไฟล์สถิติสรุปเป็นแผนภูมิภาพที่ [speedup_benchmark.png](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/speedup_benchmark.png) ให้คุณได้รับชมทันทีครับ!
