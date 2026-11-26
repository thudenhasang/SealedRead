# 07 · Kế hoạch thử nghiệm trên Colab Pro

**Mục tiêu.** Colab chỉ dùng để **kiểm chứng**, với các bài chạy nhỏ (D25). Trước khi dùng máy 8×B200, kiểm chứng ba điều:
1. Toàn bộ code chạy được trên GPU thật, từ đầu đến cuối, không cần sửa tay.
2. Các engine cho kết quả khớp nhau (kiểm tra 6).
3. Có số đo chi phí thật, để ước lượng lại thời gian trên máy lớn.

Colab **không** dùng để ra kết quả của bài, kể cả cổng S0. Cổng S0 chính thức chạy trên máy lớn (`pipeline.sh s0`, khoảng 1–2 giờ), sau khi commit `prereg.yaml`. Mọi con số ở đây đều trên dữ liệu thử nhỏ.

**Nguyên tắc:** chỉ dùng dữ liệu dựng từ **shard train**. Tập dev/test thật (shard s90) không được đụng tới, để chúng vẫn "chưa nhìn" khi commit `configs/prereg.yaml` (D24).

## Tổng quan
| Giai đoạn | GPU | Thời gian (ước lượng) | Lệnh | Mục đích |
|---|---|---|---|---|
| A. Kiểm tra code | CPU | 15 phút | `pytest`, `smoke.py` | code không vỡ trên môi trường Colab |
| B. Diễn tập 0.6B | G4 (hoặc L4, A100) | 1–2 giờ | `run_rehearsal.sh` | chạy hết chuỗi: dữ liệu, teacher, SFT, ReST-EM, đánh giá, báo cáo |
| C. Đo đạc 4B + kiểm tra mask + S0 thử nghiệm | **G4** (khoảng 96 GB; A100 40 GB bỏ qua teacher) | 6–7 giờ | `run_measure.sh` | tokenizer, dữ liệu thử ở độ dài thật, parity, chi phí, **kiểm tra mask trên 4B thật**, **S0 thử nghiệm ở 64K và 32K**, teacher pilot |
| D. Tổng hợp | CPU | 1 giờ | điền `docs/results/colab-test.md` | quyết định có lên máy lớn được chưa |

Tất cả các bước đều có marker trên Drive: runtime bị reset thì chạy lại đúng lệnh cũ, script tự bỏ qua các bước đã xong.

---

## Chuẩn bị (một lần)
1. Repo `thudenhasang/sealedread` là public: notebook clone được mà không cần token.
2. Shard dữ liệu: nếu Drive có `MyDrive/ragproj2/data/train/` (cùng shard keychain-dv dùng) thì script lấy từ đó; nếu không, script tự dựng lại từ Hugging Face (vài phút) và lưu một bản lên `MyDrive/sealedread/data/shards/` cho các lần sau.
3. Dữ liệu đã dựng và kết quả đều được lưu lên Drive, nên runtime bị reset thì chạy lại đúng lệnh là tiếp tục được.
4. Mở notebook, chạy các ô theo thứ tự: GPU → Drive → Clone → Setup → Restore/persist.

---

## A. Kiểm tra code (CPU, khoảng 15 phút)
```bash
cd /content/sealedread && /content/venv-srd/bin/python -m pytest -q tests
/content/venv-srd/bin/python scripts/smoke.py --out /content/runs/smoke
```
**Kết quả đạt:**
- `96 passed`;
- `SMOKE_DONE` (đủ 7 bước).

**Nếu trượt:** khác biệt phiên bản thư viện trên Colab. Gửi log để sửa trước khi dùng GPU.

---

## B. Diễn tập toàn chuỗi với Qwen3-0.6B (L4, khoảng 1–2 giờ)
```python
bg('rehearsal', 'bash scripts/colab/run_rehearsal.sh')
```
Chạy với 120 câu train, ngữ cảnh 4–8K; test 40 câu ở 8K và 16K. "Teacher" là chính 0.6B; nếu nó tạo quá ít trace, script tự bù bằng trace giả lập.

| Bước (marker) | Kiểm tra cái gì | Kết quả đạt |
|---|---|---|
| `reh_build`, `reh_check` | dựng dữ liệu bằng tokenizer Qwen3 thật; kiểm tra 1/3/4/5 | `BUILD_DONE`; `checks.json` (bỏ qua AUC nếu mẫu quá ít) |
| `reh_teacher`, `reh_teacher_val` | giao thức hai lượt gọi trên vLLM; ngữ pháp `<locate>` (structured output); bộ lọc trace | `TEACHER_DONE`; log ghi số trace thật của teacher |
| `reh_sft_sealedread`, `reh_sft_nomask` | trainer trên GPU: FlexAttention compile, LoRA, checkpoint | `TRAIN_DONE`; `best/adapter/` tồn tại; loss giảm |
| `reh_rest_*` | vLLM nạp adapter LoRA; sinh N = 4 mẫu; chọn lọc | `REST_SELECT_DONE`; `selected_soft.jsonl` |
| `reh_mstep_*` | M-step chạy tiếp từ checkpoint S1 (`init_from`) | `TRAIN_DONE` |
| `reh_eval_*` | đánh giá hai nhánh; bản tóm tắt | `EVAL_DONE`; các file `*.summary.json` |
| cuối | `report.py compare` | in ra Δ (con số không có ý nghĩa) |

**Kết quả đạt của giai đoạn B:** có `REHEARSAL_DONE`, và mọi lỗi phát sinh đã được sửa trong code và commit lên GitHub. Con số EM ở đây vô nghĩa (mô hình 0.6B, rất ít trace).

**Ghi lại:**
- số trace thật mà teacher 0.6B tạo được (cho biết mô hình nhỏ tuân thủ giao thức đến đâu);
- thời gian từng bước;
- mọi lỗi đã gặp.

---

## C. Đo đạc với Qwen3-4B, kiểm tra mask và S0 thử nghiệm (G4, khoảng 6–7 giờ)
```python
bg('measure', 'bash scripts/colab/run_measure.sh')
```
Dữ liệu thử dựng từ shard train ở **độ dài thật**, cỡ nhỏ:
- 60 câu train 16–24K;
- 20 câu val;
- dev: **150 câu ở 32K, 300 câu ở 64K** (cho S0 thử nghiệm), 30 câu ở 128K;
- test: 50 câu ở 64K và bản hư cấu của chúng.

| Bước | Đo / kiểm tra | Kết quả đạt (ngưỡng ở `prereg.yaml`) | Ghi lại |
|---|---|---|---|
| `m_tokenizer` | nhãn `[[S_k]]` prefix-free; bản sao `<locate>` dài bằng bản gốc; thẻ không phải special token | `TOKENIZER_CHECK_PASS` | số token tối đa của một khối `<locate>` (phải ≤ 48) |
| `m_build` + kiểm tra dữ liệu | dữ liệu thật ở 32K–128K | kiểm tra 1 ≥ 99%; **AUC bề mặt ≤ 0,60**; không rò ≥ 99% | AUC của tập thường và tập hư cấu; thời gian dựng |
| `m_closed_book` | kiểm tra 2 | closed-book ≤ 15% (multi-hop), ≤ 2% (hư cấu) | hai con số; nếu multi-hop vượt ngưỡng → tập hư cấu thành tập chính của RQ1 |
| `m_parity` | **kiểm tra 6**: vLLM so với mô hình tự cài; FlexAttention so với mask dense | `PARITY_PASS`: trung vị ≤ 1,5 lần mốc, p99 ≤ 2 lần mốc | trung vị và p99 của từng cặp |
| `m_cost_64k`, `m_cost_128k` | tốc độ engine hai luồng (SealedRead) và một luồng (Không mask); LoRA trong vLLM có tác dụng | `COST_PROBE_DONE`; `lora_changes_output: true` | giây mỗi episode, số vòng trung bình, ở 64K và 128K |
| `m_cost_sft_*` | 5 bước SFT 4B LoRA ở 16–24K, hai nhánh | chạy được, không hết bộ nhớ | giây mỗi bước (4 trace), VRAM đỉnh |
| `m_overfit_sealedread`, `m_overfit_nomask` | **overfit 20 trace** với 4B (8 epoch): loss phải tiến về gần 0 | `OVERFIT_PASS` (loss cuối ≤ 0,3, hoặc ≤ 20% loss đầu) | loss đầu, loss cuối. Trượt nghĩa là target hoặc mask sai |
| `m_probes_sealedread`, `m_probes_nomask` | **phép thử bất biến và phụ thuộc** trên 4B thật (20 trace) | SealedRead: `INVARIANCE_PASS`; Không mask: KL > 0 (mức rò) | KL lớn nhất, nhiễu số học, độ sụt log-prob khi đổi đáp án trong đoạn dán |
| `m_equiv` | chuỗi có mask so với chuỗi ngắn nhảy vị trí (cơ sở của cấu hình (b)) | `EQUIV_PASS` (p99 \|Δlog p\| ≤ max(2 × nhiễu, 0,02)) | p99 và nhiễu |
| `m_p0_vllm`, `m_p0_ref`, `m_recall` | **S0 thử nghiệm** (D26): ở 64K (d), (f) 300 câu, (a) 100, (b) 150; ở 32K mỗi cấu hình 150 câu; recall của `<locate>` zero-shot 100 câu | `P0_DONE`, `P0_RECALL_DONE`; `s0/p0_pilot.json` (64K) và `s0/p0_pilot_32k.json` có mục `pilot_reading` | EM(f) − EM(d) kèm CI ở từng mốc; Π = EM(f) − EM(b); EM(f) − EM(a); Recall@3; tỉ lệ `format_fixes` |
| `m_teacher_pilot` (GPU ≥ 70 GB) | teacher 27B FP8 trên 100 câu (G4) hoặc 30 câu (A100 80 GB) | `TEACHER_PILOT_DONE` | trần token p99 từng thẻ; tỉ lệ qua lọc; số giây mỗi episode và **ước lượng thời gian sinh 2.000 / 3.000 trace** trên GPU này |

**Kết quả đạt của giai đoạn C:** có `MEASURE_DONE`, `TOKENIZER_CHECK_PASS`, `PARITY_PASS`, `OVERFIT_PASS` (cả hai nhánh), `INVARIANCE_PASS`, `EQUIV_PASS`, các kiểm tra dữ liệu đạt (hoặc đã hiểu rõ lý do trượt), và có đủ số đo chi phí.

### Đọc S0 thử nghiệm: có nên làm tiếp không
S0 thử nghiệm dùng câu hỏi từ shard train, chỉ ở 64K, nên **không thay cổng chính thức**. Nó cho tín hiệu sớm, đọc theo `pilot_reading` trong `s0/p0_pilot.json` (ngưỡng "rõ" = 5 điểm):

| EM(f) − EM(d) | Đọc là | Làm gì |
|---|---|---|
| ≥ +5 và cận dưới CI > 0 | dương rõ: che có ích khi bằng chứng đã đúng | làm tiếp, lên máy lớn |
| dương nhưng nhỏ, hoặc CI chứa 0 | chưa kết luận được | vẫn làm tiếp; S0 chính thức trên máy lớn (1.000 câu, 64K và 128K) sẽ quyết định |
| ≤ −5 và cận trên CI < 0 | âm rõ | **dừng lại xem trace trước khi lên máy lớn**: mô hình có theo định dạng không (`format_fixes`), có dùng đoạn dán không, Q' có ở đúng chỗ không. Nếu không phải lỗi kỹ thuật thì chuẩn bị theo kịch bản dự phòng |

Lưu ý: đây là phép đo **zero-shot**. Mô hình chưa huấn luyện chưa quen với việc bị che, nên tín hiệu âm ở đây chưa chắc nghĩa là sau huấn luyện cũng âm. Câu trả lời thật cho giả thuyết nằm ở Δ_mask sau SFT, trên máy lớn.

Nếu sau S0 thử nghiệm có sửa prompt hay định dạng, ghi lại trong `decisions.md` **trước khi** commit `prereg.yaml`.

---

## D. Tổng hợp (khoảng 1 giờ)
Điền `docs/results/colab-test.md` (mẫu có sẵn) bằng các số đã ghi. Sau đó:
1. **Ước lượng lại thời gian trên máy lớn** từ số đo:
   - đánh giá ≈ (giây mỗi episode ở 64K/128K) × số episode của Bảng 8;
   - SFT ≈ (giây mỗi bước) × số bước.
   Nhớ quy đổi từ A100 sang B200: B200 nhanh hơn khoảng 2–3 lần; con số chính xác đo lại trên 1 GPU B200.
2. **Cập nhật** trần token (từ teacher pilot, nếu có) trong `configs/base.yaml` → `protocol.caps`.
3. **Commit** các sửa lỗi, rồi commit `configs/prereg.yaml` và ghi hash vào D0.

## Sang máy 8×B200: ngày đầu chạy `bringup.sh`
Colab chỉ có một GPU, và là dòng khác B200, nên ngày đầu trên máy lớn chạy `bash scripts/cluster/bringup.sh` (khoảng 1–2 giờ): test, tokenizer, parity và chi phí trên 1 GPU B200; rồi thử nhiều GPU ở quy mô tí hon (`par.sh` 2 phần, torchrun SFT trên 8 GPU, teacher 4 tiến trình × 2 GPU rồi `--merge`). Có `BRINGUP_PASS` thì mới chạy `pipeline.sh s0`. Cài đúng phiên bản vLLM / torch đã chạy được trên Colab (`logs/venv-srd.freeze.txt` trên Drive; `VLLM_SPEC=vllm==<phiên bản>`).

## Tiêu chí để lên máy 8×B200
- [ ] A, B, C chạy hết bằng đúng các lệnh trên, không cần can thiệp tay (mọi lỗi đã sửa và commit).
- [ ] `PARITY_PASS`, `OVERFIT_PASS`, `INVARIANCE_PASS`, `EQUIV_PASS`.
- [ ] `TOKENIZER_CHECK_PASS`; kiểm tra dữ liệu 1, 3, 4 đạt trên dữ liệu thật.
- [ ] Closed-book đã đo; biết tập nào là tập chính của RQ1.
- [ ] Có ước lượng thời gian máy lớn từ số đo, và con số đó nằm trong thời gian được phép dùng máy.
- [ ] `prereg.yaml` đã commit, hash ghi ở D0.

## Lỗi hay gặp và cách xử lý
| Hiện tượng | Nguyên nhân thường gặp | Xử lý |
|---|---|---|
| vLLM báo lỗi `structured_outputs` / `guided_decoding` | API đổi tên giữa các phiên bản | `engine/vllm_engine._structured` thử cả hai tên; nếu vẫn lỗi, ghi phiên bản vLLM và gửi log |
| Hết bộ nhớ khi chạy 128K trên A100 40 GB | KV của một câu 128K khoảng 19 GB | giảm `engine.gpu_memory_utilization` hoặc `engine.max_model_len`; chạy 128K riêng |
| Lần SFT đầu rất chậm | FlexAttention compile cho mỗi độ dài (đã đệm lên bội số của 1024) | bình thường ở vài bước đầu; ghi riêng thời gian compile |
| `ninja` / compile lỗi | PATH thiếu venv | `common.sh` đã thêm venv vào PATH (bài học từ keychain-dv) |
| Teacher 0.6B ra quá ít trace | mô hình nhỏ chưa theo được giao thức | script tự bù trace giả lập; ghi lại tỉ lệ thật |
| Runtime bị reset | giới hạn phiên Colab | chạy lại đúng lệnh; marker bỏ qua các bước đã xong |

## Chi phí Colab (ước lượng)
- **B:** khoảng 1–2 giờ L4.
- **C:** khoảng 6–7 giờ (trong đó khoảng 2,5 giờ cho S0 thử nghiệm, khoảng 1 giờ cho kiểm tra mask, 0,5–1 giờ cho teacher pilot).
- Trên **G4** (khoảng 9 CU/giờ): A + B + C khoảng 8–10 giờ, tức khoảng 75–90 CU; cộng dự phòng sửa lỗi thì khoảng 100–130 CU.

Với gói Pro, tổng số unit nằm trong mức một tháng. Xem lượng tiêu thụ thực tế trong bảng **Resources** của Colab; thiếu thì mua thêm unit. Mức giá có thể đã thay đổi so với hiểu biết của tôi.
