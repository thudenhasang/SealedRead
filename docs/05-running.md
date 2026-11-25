# 05 · Chạy ở đâu, chạy thế nào

## Máy cá nhân (CPU)
```bash
python -m venv .venv && .venv/bin/pip install -e ".[train,dev]"
.venv/bin/python -m pytest -q          # 94 test, khoảng 1 phút
.venv/bin/python scripts/smoke.py      # chạy thử cả chuỗi trên dữ liệu tổng hợp với mô hình tí hon
```
Dữ liệu thật có thể dựng trên CPU, đếm token xấp xỉ:
```bash
python scripts/build_data.py --config configs/data.yaml --shards-dir <shards> --out data/srd build.tokenizer=approx
```

## Colab (A100)
Dùng [notebooks/colab_quickstart.ipynb](../notebooks/colab_quickstart.ipynb), hoặc cầu nối `tools/colab.py` như với keychain-dv. Các script dùng lại cùng quy ước với repo cũ:
- marker trên Drive, chạy lại thì bỏ qua bước đã xong;
- `persist.sh` đồng bộ lên Drive mỗi 5 phút;
- `run_sft.sh` dừng sạch trước khi hết phiên 24 giờ (`TIME_CAP`), phiên sau chạy tiếp từ checkpoint.

```
/MyDrive/ragproj2/data/train/      shard gốc (chỉ đọc)
/MyDrive/sealedread/
  data/srd/                        mẫu đã dựng + manifest + checks.json
  runs/<run>/                      config.yaml, metrics.jsonl, ckpt/step_XXXXX/, best/adapter/
  runs/s0/, runs/teacher/, runs/eval/
  logs/, markers/
```
Đường dẫn trong config là `${DATA:-/content/data/srd}` và `${RUNS:-/content/runs}`. `common.sh` đặt hai biến này.

**Giới hạn GPU trên Colab:** teacher 27B FP8 không vừa một A100 khi cần ngữ cảnh 16K–24K với nhiều mẫu song song. Trên Colab, pilot teacher nên dùng mô hình nhỏ hơn (sửa `engine.model` khi gọi), còn phần teacher thật chạy trên máy lớn.

## Máy 8×B200
```bash
export WORK=/data/sealedread                     # đĩa nhanh
bash scripts/cluster/setup.sh                    # venv: vLLM + repo; chạy test
python scripts/fetch_data.py --from-hf --out $WORK/shards
python scripts/build_data.py --config configs/data.yaml --shards-dir $WORK/shards --out $WORK/data/srd
bash scripts/cluster/pipeline.sh s0              # rồi s1a, s1b, s2, s4, s5 sau khi đọc các quyết định
```
- `par.sh <tên> <script> <config> [overrides]`: chạy một script trên mọi GPU, mỗi GPU một phần câu hỏi.
  Với `GPUS_PER_PROC=2`, mỗi tiến trình nhận 2 GPU (teacher 27B, tensor parallel 2): 4 tiến trình song song.
  Sau đó gộp bằng `teacher_traces.py ... --merge`.
- `train.sh <config> [overrides]`: torchrun trên mọi GPU.

**Phiên bản:** ghi `uv pip freeze` vào `logs/`. Pin phiên bản vLLM đã chạy được ở bước đo chi phí (`VLLM_SPEC=vllm==x.y.z`). API structured output của vLLM có đổi tên giữa các bản; `vllm_engine._structured` thử cả hai tên.
