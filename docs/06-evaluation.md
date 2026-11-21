# 06 · Đánh giá

## Chỉ số (`eval/metrics.py`, mỗi episode một bản ghi)
| Nhóm | Chỉ số |
|---|---|
| Trả lời | EM (F1 tập hợp cho câu liệt kê), token-F1 |
| Định vị | precision / recall của mọi ID đã `<locate>` so với G; "đúng mà không nhìn" (recall 0, trả lời đúng) |
| Độ kín | KL và log-prob của phép thử bất biến; độ sụt log-prob của phép thử phụ thuộc (`eval/probes.py`) |
| Kênh ID, quy nguồn | tỉ lệ nhảy bước (dựa trên văn bản pha che và văn bản đã dán); UCR (đáp án đúng, dạng mở, không xuất hiện trong văn bản đã dán); độ chính xác khi hoán vị ID |
| Hành vi | số vòng, tỉ lệ chạm R_max, token theo thẻ, Acc, tỉ lệ mở ngữ cảnh khi \|G\| = 0 |

"Đúng mà không nhìn" và UCR được tách theo đáp án mở / đóng.

## Thống kê (`eval/stats.py`)
- **Bootstrap ghép cặp theo cụm.** Cụm là câu hỏi gốc, gồm mọi bản diễn đạt lại và mọi mốc độ dài; với tài liệu thật thì cụm là một tài liệu. Trung bình theo seed trước khi bootstrap.
- `sample_size(0.2, 0.03)` ≈ 1.744.
- `noninferiority_n` cho thấy biên −1 điểm ở n = 1.000 thiếu lực rất nhiều (D7).

## Câu hỏi nghiên cứu và lệnh
| RQ | Chạy | Báo cáo |
|---|---|---|
| RQ1 Δ_mask | `evaluate.py eval.branch={sealedread,nomask} eval.tag=<b>_s<seed> engine.adapter=…` trên test 64K/128K/32K | `report.py rq1 --sr a,b,c --nm a,b,c --sr32 … --nm32 …` |
| RQ1 so trên cùng đoạn dán | `eval.oracle=true` cho cả hai nhánh | `report.py compare` |
| RQ1 Δ_ctx | `eval.branch=daft engine.backend=ref` (SFT) và `sealedread` (SFT) | `report.py compare` |
| RQ2 với RAG | `fetch_benchmarks.py`; `eval.branch=rag` (multi-round, đã huấn luyện), `rag1` (một lượt); `sealedread` | `report.py rq2 --key f1` |
| RQ2 hoán đổi bộ chọn | `eval.branch=sealedread` với `locate` thay bằng retriever: chạy nhánh `rag` bằng adapter của SealedRead; oracle G: `eval.oracle=true` | `report.py compare` |
| RQ3 cổng thưởng | `pipeline.sh s4` (lọc lại soft / hard / nogate rồi chạy M-step) | `report.py compare` và "đúng mà không nhìn" |
| RQ4 (a) kiểm thử cài đặt | `probes.py --branch sealedread` | `invariant_rate`: phải bằng 1 |
| RQ4 (b) mức rò | `probes.py --branch nomask/daft`; `eval.hide_locate=false engine.backend=ref` | KL |
| RQ4 (c) kênh ID | `eval.permute_ids=true`; tỉ lệ nhảy bước trong mọi bản tóm tắt | `report.py compare` |
| RQ4 (d) quy nguồn | UCR trong bản tóm tắt (tập hư cấu kỳ vọng ≈ 0) | `report.py summary` |
| RQ5 tổng quát hóa | Δ_mask theo từng mốc 32K/64K/128K; NoLiMa, BABILong (file mẫu theo định dạng `Sample`) | `report.py rq1` theo từng file |

**Baseline:**
- full attention zero-shot: `eval.branch=plain`;
- chép-rồi-giải: `copy`;
- SEALONG: `plain` với adapter `sft_sealong`;
- DA zero-shot: nhánh `daft` với engine tham chiếu, không adapter.

**Ablation pha suy luận** (checkpoint SealedRead seed 1, 128K, 500 câu):
- `eval.repeat_q=false` (chỉ Q trước C);
- `prompts.show_titles=false`;
- `eval.locate_query_only=true`;
- `eval.hide_locate=false engine.backend=ref`.

## Engine nào chạy được gì
| Cần | vLLM (nhanh) | Engine tham chiếu (chậm, chính xác) |
|---|---|---|
| SealedRead hai luồng, Không mask, RAG, plain | có | có (trừ plain) |
| Bố cục chính một chuỗi (khung đầy đủ), DA-ft, tắt quy tắc ẩn | không | có |
| Phép thử bất biến / phụ thuộc | không | có (FlexAttention trên GPU) |
