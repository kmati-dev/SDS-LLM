# Plan: Tensor-based n-gram Drafter + Verifier (depth & width)

## Context

โปรเจคนี้เป็น **simulator** ของ greedy speculative decoding ที่ไม่รันโมเดลจริง — ใช้ ground
truth tokens แทน target model เพื่อวัด speedup เร็ว ๆ บน CPU. ปัจจุบันทุกอย่างเป็น Python `List[int]`:
- [NGramDrafter](src/simulator.py#L5) คืน 1 sequence (`List[int]`)
- [GreedyVerifier](src/simulator.py#L47) รับ 1 sequence
- ทดสอบด้วย pytest 33 เคสใน [tests/test_interfaces.py](tests/test_interfaces.py)

เป้าหมายของงานนี้ (ส่วนที่ทำตอนนี้): ทำให้ drafter draft token ออกมาเป็น **tensor** และ verifier
รับ tensor ที่อาจมี **หลาย sequence** ได้ เพื่อปูทางไปเทียบ **depth-draft** กับ **width-draft**:
- **depth-draft**: budget B token → 1 sequence ยาว B → tensor shape `[1, B]`
- **width-draft**: budget B token → S sequence ยาว T (S·T ≤ B) → tensor shape `[S, T]`
- **verifier**: ถ้ามีหลาย sequence ต้องตรวจทุกสาย แล้วเลือกสายที่ match เฉลยได้ยาวที่สุด

**Decisions (ยืนยันกับ user):** ใช้ **PyTorch**; **เพิ่ม class/method ใหม่ คงของเดิมไว้ทั้งหมด**
(test เดิม 33 เคสต้องไม่พัง); **dataset + การเทียบ depth-vs-width จริง = เก็บไว้ทำทีหลัง** (ดู Out of scope).

## Tensor representation (ข้อตกลงร่วม)

- draft = `torch.Tensor` dtype `torch.long`, shape `[S, T]` (2D เสมอ — depth คือ S=1).
- ถ้า candidate บางสายสั้นกว่า T (เจอใกล้ปลาย corpus) → pad ด้วย `-1` (PAD sentinel), verifier ข้าม `-1`.
- ไม่เจอ match เลย → คืน tensor ว่าง shape `[0, 0]` (หรือ `[0, T]`); playback/verifier ถือเป็น "no draft".

## Changes

### 1. Tensor interfaces — [src/interfaces.py](src/interfaces.py)
เพิ่ม abstract class ใหม่ (ไม่แตะของเดิม):
- `AbstractTensorDrafter.generate_draft(self, prompt: List[int]) -> "torch.Tensor"` (shape `[S, T]`)
- `AbstractTensorVerifier.verify(self, draft_tokens: "torch.Tensor", current_prefix: List[int], complete_tokens: List[int]) -> Dict[str, Any]`

### 2. Tensor drafter — [src/simulator.py](src/simulator.py)
เพิ่ม `TensorNGramDrafter(AbstractTensorDrafter)` — class เดียวคุม depth/width ผ่านพารามิเตอร์:
```python
def __init__(self, corpus_tokens, n=3, num_sequences=1, draft_depth=3): ...
```
- `num_sequences=S`, `draft_depth=T`; budget = S·T. depth = `(S=1, T=B)`, width = `(S>1, T<B)`.
- ใช้ตรรกะ n-gram + backoff เดิมจาก [generate_draft](src/simulator.py#L18) เป็นฐาน แต่:
  - **เก็บหลาย match**: หา match ของ (n-1)-gram ทุกตำแหน่งใน corpus (backoff ลงไป 1-gram เหมือนเดิม),
    ดึง continuation ยาว T ต่อ match.
  - **dedupe + กระจาย branch**: เก็บ continuation ที่ไม่ซ้ำ โดยให้ความสำคัญกับ "first token ต่างกัน"
    ก่อน (นี่คือหัวใจของ width = hedge ที่ branch point) แล้วตัดเอาแค่ S สาย.
  - **pad** สายที่สั้นกว่า T ด้วย `-1`, stack เป็น tensor `[S, T]` dtype long.
  - คงตัวแปร metric `last_n_used` / `last_match_corpus_idx` ให้ playback อ่านได้.

### 3. Tensor verifier — [src/simulator.py](src/simulator.py)
เพิ่ม `TensorGreedyVerifier(AbstractTensorVerifier)`:
- รับ draft `[S, T]` (อาจมี `-1` pad). สำหรับแต่ละแถว s: เทียบทีละ token กับ
  `complete_tokens[prefix_len + i]` หยุดเมื่อ mismatch/เจอ pad/เกินความยาวเฉลย → ได้ matched length `L_s`.
- เลือกแถวที่ `L_s` มากสุด (tie-break: index ต่ำสุด).
- คืน dict สอดคล้องกับ verifier เดิม + ฟิลด์ใหม่:
  ```python
  {"accepted_tokens": [...best row[:L] + recovery token...],
   "accepted_count": L, "rejected_count": <remaining in chosen row>,
   "chosen_sequence": <row idx>}
  ```
- recovery token = `complete_tokens[prefix_len + L]` ถ้ายังไม่จบเฉลย (เหมือน [GreedyVerifier](src/simulator.py#L71)).

### 4. Dependency — [pyproject.toml](pyproject.toml)
เพิ่ม `torch` ใน `[project].dependencies`. ติดตั้งด้วย `pip install torch` (ยังไม่มีในเครื่อง).

## Tests — `tests/test_tensor_interfaces.py` (ไฟล์ใหม่)
สไตล์เดียวกับ [tests/test_interfaces.py](tests/test_interfaces.py) (docstring ต่อเคส, corpus เลขเล็ก ๆ):
- **abstract**: instantiate `AbstractTensorDrafter`/`AbstractTensorVerifier` ตรง ๆ ต้อง TypeError.
- **TensorNGramDrafter (depth, S=1)**: perfect match → shape `[1,T]`, dtype long, ค่าถูก; backoff; no-match → `[0,*]`.
- **TensorNGramDrafter (width, S>1)**: คืน `[S,T]`; เก็บหลาย continuation ที่ first-token ต่างกัน; dedupe; pad `-1` เมื่อสายสั้น; เจอ candidate < S → คืนเท่าที่มี.
- **TensorGreedyVerifier**: ทุกแถว accept/reject; partial; เลือกสายที่ยาวสุดเมื่อหลายสาย; tie-break index ต่ำสุด; ข้าม pad `-1`; recovery token; ขอบเขตปลายเฉลย; `chosen_sequence` ถูกต้อง.
- **shape/dtype invariants**: draft เป็น 2D long เสมอ.

## Verification
1. `pip install torch` แล้ว `python3 -m pytest tests/ -v` → เคสเดิม 33 + เคสใหม่ผ่านทั้งหมด.
2. Smoke check ใน REPL: สร้าง `TensorNGramDrafter` ทั้งโหมด depth `(S=1,T=10)` และ width `(S=5,T=2)`
   บน corpus เล็ก ตรวจ shape `[1,10]` / `[5,2]`, ส่งผ่าน `TensorGreedyVerifier` ดูว่าเลือกสายถูก.

## Out of scope (เก็บไว้ทำทีหลังตามที่ user ระบุ)
- เพิ่ม codegen dataset (MBPP/HumanEval/CodeSearchNet) เข้า [src/datasets](src/datasets/__init__.py).
- `TensorSpeculativePlayback` + เดินสาย benchmark ใน [run.py](run.py) เพื่อเทียบ speedup depth vs width จริง.
