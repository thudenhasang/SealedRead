# 04 · Kế hoạch chạy

> Mọi ngưỡng ở `configs/prereg.yaml`. Commit file này và ghi hash vào `decisions.md` **trước** S0.

## Trước khi thuê máy lớn (Colab A100)
| Bước | Lệnh | Điều kiện đi tiếp |
|---|---|---|
| Test CPU | `pytest -q`, `python scripts/smoke.py` | 91 test qua, `SMOKE_DONE` |
| Diễn tập 0.6B (giai đoạn 7 của kế hoạch) | `bash scripts/colab/run_rehearsal.sh` | mọi bước chạy hết không sửa tay; `REHEARSAL_DONE` |
| Dữ liệu thật (M1) | `bash scripts/colab/run_data.sh` | `TOKENIZER_CHECK_PASS`, `DATA_CHECKS_PASS` |
| S0 trên 4B | `bash scripts/colab/run_s0.sh` | parity đạt (kiểm tra 6); có số đo chi phí; quyết định P0 |
| Chi phí SFT | `python scripts/cost_probe.py configs/sft.yaml --part sft --steps 5` | s/bước, VRAM đỉnh → cập nhật Bảng 7 |

## Trên máy 8×B200 (`scripts/cluster/pipeline.sh <giai đoạn>`)
| Giai đoạn | Việc | Lệnh / script | Quyết định chốt trước |
|---|---|---|---|
| S0 | 6 cấu hình P0, Recall@K, closed-book, pilot teacher | `pipeline.sh s0`, `report.py p0` | Cổng P0 và Bảng 6 (`gates.p0_decision`) |
| S1a | trace teacher: giao thức (3.000), val, RAG, CoT | `pipeline.sh s1a` | tỉ lệ qua lọc theo nhóm, ghi trong `*.stats.json` |
| S1b | SFT LoRA: SealedRead, Không mask, DA-ft (3 seed), RAG, SEALONG; pilot full FT | `pipeline.sh s1b` | `report.py compare` LoRA và full FT (`gates.lora_vs_full`); điểm quyết định sau S1 (`report.py s1`, dev 64K); cổng S2 (`report.py s2`) |
| S2 | một vòng ReST-EM cho SealedRead, Không mask, RAG; M-step 3 seed | `pipeline.sh s2` | vòng 2 tùy chọn (`report.py round2`) |
| S4 | RQ3: lọc lại và chạy M-step | `pipeline.sh s4` | |
| S5 | đánh giá (Bảng 8) | `pipeline.sh s5` + `docs/06-evaluation.md` | RQ1 (`report.py rq1`), RQ2 |

**Ghi chú về cách chạy:**
- Mỗi GPU chạy một tiến trình vLLM với một phần câu hỏi (`par.sh`). File kết quả tên `*.shard{i}of{n}.jsonl`, và `report.py` / `rest_em.py --select-only` gom lại bằng glob.
- SFT chạy trên mọi GPU bằng torchrun (`train.sh`). Gradient của LoRA được all-reduce trong trainer.
- Teacher 27B FP8 chạy tensor parallel 2 (`configs/teacher.yaml`). Tên checkpoint chính xác cần kiểm tra trên Hugging Face (D16).

## Siêu tham số (theo đề xuất)
| Mục | Giá trị | Ở đâu |
|---|---|---|
| Giao thức | K = 3, R_max = 8, trần token theo p99 của pilot teacher | `configs/base.yaml`, `scripts/teacher_traces.py --pilot` |
| Thưởng | λ = 0,2, μ = μ₀ = 0,25 | `configs/base.yaml` |
| SFT | LoRA r = 64, α = 128 trên mọi lớp tuyến tính; lr 1e-4; 2–3 epoch; dừng sớm theo loss validation | `configs/sft.yaml` |
| ReST-EM | N = 4, T = 0,7, lọc cứng, giữ ≤ 2 mỗi câu, trần 4.000 | `configs/rest_em.yaml` |
| Seed | 3 seed của M-step (cùng tập mẫu đã sinh) | `pipeline.sh` |

## Những gì chưa được chạy
Phần GPU mới được viết, chưa chạy lần nào. Kinh nghiệm từ keychain-dv: lần chạy GPU đầu tiên lộ ra 5 lỗi. Vì vậy diễn tập 0.6B phải xong trước khi đụng tới 4B hay máy lớn.
