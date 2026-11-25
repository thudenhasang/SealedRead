# Tài liệu

| Trang | Nội dung |
|---|---|
| [01-design.md](01-design.md) | Thiết kế tóm tắt: ý tưởng, cài đặt hai luồng, các nhánh so sánh, Tính chất 1 trong code |
| [02-data.md](02-data.md) | Nguồn dữ liệu, cách dựng mẫu, 5 loại câu, tách tập, kiểm tra dữ liệu |
| [03-protocol.md](03-protocol.md) | Thẻ, máy trạng thái, tokenization chuẩn, bố cục token, mask và position ID của từng nhánh |
| [04-training-plan.md](04-training-plan.md) | Các giai đoạn S0–S5: lệnh, cổng quyết định, chạy ở đâu |
| [05-running.md](05-running.md) | Chạy trên máy cá nhân, Colab và máy 8×B200 |
| [06-evaluation.md](06-evaluation.md) | Chỉ số, RQ1–RQ5, phép thử, ablation, lệnh báo cáo |
| [07-colab-test-plan.md](07-colab-test-plan.md) | **Kế hoạch thử nghiệm trên Colab Pro** và tiêu chí để lên máy lớn; ghi kết quả vào [results/colab-test.md](results/colab-test.md) |
| [decisions.md](decisions.md) | Nhật ký quyết định, gồm các chỗ sửa so với bản đề xuất 2.4 |

## Từ đề xuất đến code

| Mục trong đề xuất | Code | Test |
|---|---|---|
| 3.1 Cấu trúc đầu vào (bố cục kẹp, chia đoạn, ID ngẫu nhiên) | `prompts.py`, `data/chunk.py`, `data/ids.py`, `layout.add_prompt` | `test_layout`, `test_data` |
| 3.2 Chu trình sáu thẻ, máy trạng thái, trần token | `protocol.py`, `engine/driver.py` | `test_protocol`, `test_engine` |
| 3.3 Mask (pt. 1), position ID, Acc (pt. 2) | `layout.visible`, `model/attention.make_mask_mod`, `reward.acc_cost` | `test_layout`, `test_model` |
| 3.4 `<locate>` + engine dán | `grammar.py`, `driver._locate`, `prompts.knowledge_body` | `test_protocol` |
| 3.5 Bố cục kẹp, chẩn đoán vị trí | view `masked` với frame `full`/`compact` | `test_layout` |
| 3.6 Kênh rò, Tính chất 1, phép thử phản thực tế | `eval/probes.py` | `test_model`, `test_probes_bench` |
| 3.7 Một chuỗi hay hai luồng | `engine/ref.py` (một chuỗi), `engine/vllm_policy.py` (hai luồng) | `test_engine` |
| 3.8 Backbone, engine | `model/qwen3.py`, `engine/vllm_engine.py`, `train/sft.py` | `test_model` |
| 4 Dữ liệu, kiểm tra 1–6 | `data/*`, `scripts/check_data.py`, `scripts/parity.py` | `test_data` |
| 5.1 Thưởng R_μ | `reward.py` | `test_reward` (Bảng 5) |
| 5.2 S0 (6 cấu hình, cổng P0, Bảng 6) | `scripts/p0_probe.py`, `eval/gates.p0_decision` | `test_train_eval` |
| 5.3 Teacher, SFT | `teacher.py`, `train/data.py`, `train/sft.py` | `test_train_eval` |
| 5.4 ReST-EM | `rest_em.py`, `scripts/rest_em.py` | `test_train_eval` |
| 6 RQ1–RQ5, thống kê | `eval/metrics.py`, `eval/stats.py`, `eval/gates.py`, `scripts/report.py` | `test_train_eval` |
| Phụ lục A: khung rút gọn | `frame="compact"` | `test_layout` |
| Phụ lục B: Tính chất 1 | phép thử cài đặt trong `test_model.test_invariance_*` | |

Thuật ngữ thống nhất với đề xuất:
- **pha che**: các thẻ do mô hình viết mà không thấy C;
- **bản sao**: bản sao `<locate>` mã hóa khi C bị che;
- **view**: những token nào tồn tại trong một chuỗi;
- **nhánh**: mask nào được dùng (Bảng 10).
