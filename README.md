# SealedRead: từ mask hiệu năng đến mask thông tin khi suy luận trên ngữ cảnh dài

Mã nguồn của đề xuất SealedRead (bản 2.4). Ý tưởng chính: phần suy luận (`<reason>`, `<query>`, `<recheck>`, `<answer>`) **không bao giờ nhìn thấy ngữ cảnh dài C**. Chỉ thẻ `<locate>` được nhìn C, và chỉ được xuất 1–3 ID đoạn. Sau đó engine dán nguyên văn các đoạn đó vào `<knowledge>`. Mô hình chính là Qwen3-4B-Instruct-2507, huấn luyện bằng SFT LoRA từ trace của teacher, rồi thêm một vòng ReST-EM.

## Trạng thái (2026-11-21)

| Phần | Trạng thái |
|---|---|
| Lõi giao thức: 6 thẻ, máy trạng thái, ngữ pháp `<locate>`, engine dán, quy tắc hợp lệ | Xong, có test CPU |
| Bố cục token và mask của mọi nhánh (SealedRead, Không mask, DA-ft), 3 view, vị trí | Xong. Test từng ô của Hình 1b |
| Mô hình Qwen3 tự cài (mask tường minh, position ID tùy chỉnh, KV cache), LoRA định dạng PEFT | Xong. Khớp `transformers` trên trọng số ngẫu nhiên |
| Engine tham chiếu | Xong. Trace nó sinh ra cho **đúng từng token và mask** như trainer dựng |
| Tính chất 1 như phép thử cài đặt | Xong. SealedRead bất biến tuyệt đối; Không mask và DA-ft rò rỉ đúng như Bảng 2 |
| Thưởng R_μ (pt. 2, 4, 5) | Xong. Test lại từng dòng của Bảng 5 |
| Dữ liệu (M1): chia đoạn, ID ngẫu nhiên, nhiễu BM25, lọc rò, thực thể hư cấu, 5 loại câu, kiểm tra 1/3/4/5 | Xong, có test CPU trên bản ghi tổng hợp đúng định dạng gốc |
| SFT (mọi nhánh), ReST-EM (chọn lọc, RQ3), teacher (lọc, hạn ngạch), đánh giá, bootstrap, các cổng chốt trước, phép thử mask | Xong, có test CPU. Chạy thử toàn chuỗi trên CPU: `scripts/smoke.py` |
| Engine nhanh trên vLLM (hai luồng / một luồng), script GPU (S0, teacher, parity, đo chi phí, đánh giá, benchmark) | **Đã viết, chưa chạy trên GPU.** Lần chạy đầu: `scripts/colab/run_rehearsal.sh` (Qwen3-0.6B trên A100) |
| Engine một chuỗi trên vLLM (bố cục chính ở tốc độ cao) | Chưa làm: spike E0. Hiện chạy bằng engine tham chiếu (chậm), xem [docs/decisions.md](docs/decisions.md) D1 |

## Bắt đầu nhanh

**Máy cá nhân (chỉ CPU):** test và chạy thử toàn chuỗi trên dữ liệu tổng hợp với mô hình tí hon, khoảng 2 phút.
```bash
python -m venv .venv && .venv/bin/pip install -e ".[train,dev]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke.py
```

**Colab (A100):** mở [notebooks/colab_quickstart.ipynb](notebooks/colab_quickstart.ipynb). Thứ tự: diễn tập với Qwen3-0.6B, rồi dữ liệu thật, rồi S0. Chi tiết ở [docs/05-running.md](docs/05-running.md).

**Máy lớn (8×B200):** `scripts/cluster/` gồm `setup.sh`, `par.sh` (mỗi GPU một phần câu hỏi), `train.sh` (torchrun) và `pipeline.sh` (các giai đoạn S0 → S5).

## Cấu trúc repo
```
configs/            YAML theo giai đoạn (base + data, s0, teacher, sft, rest_em, eval, debug, benchmarks) và prereg.yaml
docs/               thiết kế, dữ liệu, giao thức & mask, kế hoạch train, cách chạy, đánh giá, nhật ký quyết định
notebooks/          notebook Colab
scripts/            mỗi bước một lệnh: fetch_data, build_data, check_data, tokenizer_check, closed_book, paraphrase,
                    p0_probe, teacher_traces, train_sft, rest_em, evaluate, probes, parity, cost_probe, report,
                    fetch_benchmarks, smoke
scripts/colab/      pipeline trên Colab (marker trên Drive, chạy tiếp sau reset, giới hạn thời gian phiên)
scripts/cluster/    pipeline trên máy nhiều GPU
src/srd/
  protocol.py       thẻ, bước, máy trạng thái, quy tắc hợp lệ
  grammar.py        ngữ pháp <locate>: trie token (engine tham chiếu), regex (vLLM)
  layout.py         bố cục token + quy tắc attend của mọi nhánh: nguồn duy nhất của mask
  prompts.py        hướng dẫn A, bố cục kẹp [A][Q][C][Q'][T']
  reward.py         R_μ, F2, Acc
  data/             qa, chunk, ids, bm25, leak, fiction, synth, build, splits, checks, paraphrase, benchmarks
  model/            qwen3 (tự cài), attention (SDPA/Flex), lora (PEFT), run (loss, log-prob)
  engine/           driver, ref (tham chiếu), vllm_engine, vllm_policy (hai luồng), retriever, plain, scripted, factory
  train/            data (view theo nhánh), sft (trainer)
  rest_em.py, teacher.py
  eval/             metrics, stats (bootstrap cụm ghép cặp), gates (cổng chốt trước), probes (bất biến/phụ thuộc), runner
tests/              test CPU (pytest), 91 test
tools/colab.py      client cầu nối Colab (giữ nguyên từ keychain-dv)
```

## Tài liệu
Bắt đầu từ [docs/README.md](docs/README.md). Thư mục này có bảng đối chiếu từng mục của đề xuất với module tương ứng.

Dữ liệu đã dựng, checkpoint và log không nằm trong git. Trên Colab chúng nằm ở `MyDrive/sealedread/`, trên máy lớn ở `$WORK`.
