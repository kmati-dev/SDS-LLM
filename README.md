# Speculate Decoding Simulator (Greedy Speculative Decoding)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

โปรเจกต์นี้ได้รับการจัดระเบียบตามแนวทางวิศวกรรมซอฟต์แวร์ (Software Engineering Guidelines) เพื่อความง่ายในการขยายระบบ ทดสอบ และใช้งาน:

```text
├── artifacts/              # สำหรับเก็บผลลัพธ์การทดสอบ (เช่น กราฟ speedup_benchmark.png)
├── configs/                # ไฟล์ตั้งค่าสำหรับระบบจำลอง
│   └── simulator_config.json
├── docs/                   # เอกสารอธิบายรายละเอียดสถาปัตยกรรมและการคำนวณ
│   └── architecture.md
├── src/                    # ซอร์สโค้ดหลักของโปรเจกต์
│   ├── interfaces.py       # สัญญาข้อตกลง (Abstract Interface Contracts)
│   └── simulator.py        # ส่วนคลาสจริง (Concrete Implementations)
├── tests/                  # สวีทการทดสอบระบบ (Automated Tests)
│   └── test_interfaces.py
├── .gitignore              # การละเว้นไฟล์ขยะใน Git
├── pyproject.toml          # กำหนด Metadata และการตั้งค่าของ pytest
├── README.md               # เอกสารแนะนำการใช้งานภาษาไทย (ไฟล์นี้)
└── run.py                  # ตัวรันการทำงานและสวีปผลประสิทธิภาพหลัก (Unified Entry Point)
```

---

## 📌 สรุปรายการ Checklist และการจับคู่ไฟล์

| รายการ Checklist | คลาส / ฟังก์ชัน ที่นำไปใช้จริง | ชื่อไฟล์หลัก | รายละเอียดเบื้องต้น |
| :--- | :--- | :--- | :--- |
| **Abstract class ของ Playback** | `AbstractPlayback` | [src/interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/src/interfaces.py) | กำหนดอินเทอร์เฟซตัวควบคุมหลัก รองรับ Dependency Injection และการรันแบบทีละโทเค็น |
| **Abstract class ของ Drafter** | `AbstractDrafter` | [src/interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/src/interfaces.py) | อินเทอร์เฟซของโมเดลผู้ร่างเพื่อเดาโทเค็นล่วงหน้า |
| **Abstract class ของ Verifier** | `AbstractVerifier` | [src/interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/src/interfaces.py) | อินเทอร์เฟซโมเดลผู้ตรวจแบบ Greedy คืนค่า Accepted และ Recovery Token |
| **Implementation n-gram decoding**| `NGramDrafter` | [src/simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/src/simulator.py) | คลาสโมเดลผู้ร่างจริงด้วยเทคนิคประวัติ N-gram และมีระบบถอยกลับ (Backoff) |
| **Implementation verifier** | `GreedyVerifier` | [src/simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/src/simulator.py) | คลาสผู้ตรวจจริง เปรียบเทียบเรียงตัวจนเจอมิสแมตช์ตัวแรกแล้วดึง Recovery Token |
| **Implementation metrics** | `PlaybackMetrics` | [src/simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/src/simulator.py) | คลาสเก็บสถิติคำนวณ ความเร็ว (Speedup), ยอดสูงสุดยอมรับ และค่าเฉลี่ยต่อขั้นตอน |
| **Implementation ของ Playback** | `SpeculativePlayback` | [src/simulator.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/src/simulator.py) | ระบบรันจำลองการทับศัพท์/ถอดโทเค็นจริงโดยดึงข้อมูลจาก Tokenizer แบบ Duck Typing |
| **Test case** | pytest unit tests | [tests/test_interfaces.py](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/tests/test_interfaces.py) | รวมเคสทดสอบ 12 เคสครอบคลุมทั้ง Abstract, Concrete, Metrics และ E2E Playback |

---

## 🧠 ลอจิกการทำงานอย่างละเอียด (Detailed Logic)

คุณสามารถศึกษาลอจิกการทำงานเชิงทฤษฎีสูตรทางคณิตศาสตร์ และคลาสไดอะแกรมอย่างเป็นทางการได้ที่ **[docs/architecture.md](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/docs/architecture.md)** 

สรุปกระบวนการสำคัญ:
1. **Abstract Interfaces (`src/interfaces.py`)**: กำหนดสัญญาร่วม ป้องกันการสร้างออบเจกต์ (Instantiation) โดยตรง
2. **N-Gram Drafter (`src/simulator.py`)**: คาดเดาคำถัดไปตามจำนวนร่างที่ตั้งไว้ ($K$) พร้อมระบบ **Backoff Mechanism** ถอยขนาด N-gram ลงมาเรื่อย ๆ เมื่อไม่พบคู่แมตช์ในประวัติ
3. **Greedy Verifier (`src/simulator.py`)**: ตรวจสอบคำร่างเปรียบเทียบกับคำจริงทีละคำ หากพบคำไม่ตรง จะปฏิเสธคำดราฟต์ที่เหลือทั้งหมด และแนบ **Recovery Token** ตัวถัดไปของคำเฉลยกลับคืนเพื่อเริ่มรอบถัดไปทันที
4. **Metrics Tracker (`src/simulator.py`)**: คำนวณอัตราเร่งความเร็ว (**Speedup Ratio**) เปรียบเทียบผลระหว่างการเรียกใช้โมเดลแบบปกติและแบบ Speculative
5. **Playback Controller (`src/simulator.py`)**: ดูแลลูปใหญ่ของการจำลองการทับศัพท์ผ่านการทำ Dependency Injection ของ Tokenizer แบบ Duck Typing

---

## 📊 ผลลัพธ์และบทวิเคราะห์ทางเทคนิค (Benchmark & Technical Analysis)

เมื่อรันไฟล์เดโมสวีป `run.py` บนระบบจำลองจะพบข้อมูลสถิติที่สำคัญดังนี้:

### 1. แหล่งที่มาของข้อมูลนำเข้า
* **คลังข้อมูลต้นแบบของ Drafter (Corpus Tokens)**: ข้อความบทความเชิงเทคนิคภาษาอังกฤษจำลองสถานการณ์ "logs ข้อความในอดีต" (Retrieval Database)
* **ข้อความเป้าหมายรันจำลอง (Target Tokens)**: ข้อความสั้นที่จำลองสถานการณ์จริงที่ LLM กำลังสร้างคำใหม่ (Target output sequence)

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

* **Baseline (1.0x)**: การรันแบบดั้งเดิมที่ต้องเรียกใช้งานโมเดลใหญ่ **134 รอบ**
* **K = 3 (จุดคุ้มค่าทางเศรษฐศาสตร์ - Sweet Spot)**: ให้ผู้ร่างเดาคำ 3 คำ ส่งผลให้ขั้นตอนรันลดลงเหลือ **50 รอบ** เร่งความเร็วขึ้นอย่างก้าวกระโดดเป็น **2.68 เท่า**
* **K = 5 (ประหยัดสูงสุด)**: ขั้นตอนรันโมเดลลดลงเหลือต่ำสุดเพียง **43 รอบ** และได้ความเร็วพุ่งสูงสุดถึง **3.12 เท่า**

---

### 3. แผนภูมิวิเคราะห์ประสิทธิภาพ (Performance Charts)

ภาพแผนภูมิความเร็วและอัตราการยอมรับคำที่สร้างขึ้นจะถูกบันทึกไว้อย่างปลอดภัยที่ **[artifacts/speedup_benchmark.png](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/artifacts/speedup_benchmark.png)**

* ** Inference Acceleration (Speedup Ratio)**: แสดงถึงความคุ้มค่าของการเพิ่มขนาดดราฟต์ $K$ ซึ่งจุดหักมุมความชันเด่นชัดจะอยู่ที่ $K=3$ จากนั้นจะเริ่มแบนราบลงตามกฎการลดน้อยถอยลงของผลตอบแทน (Law of Diminishing Returns)

---

## 🚀 วิธีการทดสอบและสั่งใช้งานจริง

### 1. วิธีรันทดสอบ Unit Tests (ผ่าน pytest)
คุณสามารถรันการทดสอบ Unit Tests ทั้ง 12 เคสครอบคลุมทุกคลาสและเมธอดได้ง่าย ๆ จากไดเรกทอรีของโปรเจกต์:
```bash
python3 -m pytest tests/test_interfaces.py -v
```

### 2. วิธีรันโปรแกรมประเมินผลประสิทธิภาพ (Centralized Sweep Benchmark)
รันไฟล์ Entry Point หลัก `run.py` เพื่อสวีปหาความเร็วและพล็อตแผนภูมิแบบพรีเมียมได้ผ่านคำสั่ง:
```bash
# รันค่าเริ่มต้น (จะโหลดพารามิเตอร์อัตโนมัติจาก configs/simulator_config.json)
python3 run.py

# หรือกำหนดพารามิเตอร์แบบกำหนดเอง (Custom Overrides)
python3 run.py --tokenizer gpt2 --n 3 --max_draft 5 --artifacts_dir artifacts
```
หลังจากประมวลผลเสร็จสิ้น ระบบจะอัปเดตไฟล์สถิติสรุปเป็นภาพแผนภูมิไปที่โฟลเดอร์ผลลัพธ์ [artifacts/speedup_benchmark.png](file:///Users/nantaporn/Documents/indiv-llm/spec-decode-greedy/artifacts/speedup_benchmark.png) โดยอัตโนมัติ
