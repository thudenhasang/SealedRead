#!/usr/bin/env bash
# Phase C of docs/07-colab-test-plan.md: a SMALL verification run on one A100 with the real 4B model (about
# 4-5 hours, of which about 2 for the pilot S0). Colab verifies code and measures costs only; the official S0 gate runs on the GPU machine
# (scripts/cluster/pipeline.sh s0) after configs/prereg.yaml is committed (D25).
# Steps: tokenizer, a small "pilot" data set
# at the real lengths (built from TRAIN shards only, D24), data checks + closed-book, engine parity (check 6),
# cost probes (vLLM two streams / one stream at 64K and 128K; SFT steps at 16-24K), a small P0 (gate (f) vs (d),
# Pi from (b)) and, on an 80 GB A100 only, a teacher pilot. Resumable through Drive markers.
#   bash scripts/colab/run_measure.sh
source "$(dirname "$0")/common.sh"
P=${PILOT:-/content/data/srd_pilot}
R=$RUNS/measure
M=Qwen/Qwen3-4B-Instruct-2507
mkdir -p "$R"
GPU_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')
log "GPU memory: ${GPU_GB} GB"
MAXLEN=$([ "$GPU_GB" -ge 70 ] && echo 140000 || echo 135000)

step fetch_shards $PY scripts/fetch_data.py --from-drive "$SHARDS" --out /content/data/shards || exit 1
step m_tokenizer $PY scripts/tokenizer_check.py $M --out "$R/tokenizer_check.json" || exit 1
step m_build $PY scripts/build_data.py --config configs/data.yaml --shards-dir /content/data/shards --out "$P" \
  "build.train_shards=[0,1]" build.eval_shard=4 build.train.n=60 build.val.n=20 build.reserve.n=5 \
  build.dev.n=300 "build.dev.lengths={32768: 20, 65536: 300, 131072: 30}" build.test.n=50 \
  "build.test.lengths={65536: 50}" "build.test.fictional_lengths=[65536]" || exit 1
step m_closed_book $PY scripts/closed_book.py configs/base.yaml --samples "$P/test_64k.jsonl" \
  "$P/test_fict_64k.jsonl" --out "$R/closed_book.jsonl" || exit 1
# data checks are reported, not blocking here: the numbers go to docs/results/colab-test.md
$PY scripts/check_data.py "$P"/*.jsonl --closed-book "$R/closed_book.jsonl" --out "$R/checks.json" \
  > "$LOGS/m_checks.log" 2>&1; tail -8 "$LOGS/m_checks.log"
step m_scripted $PY scripts/make_scripted_traces.py --samples "$P/train.jsonl" --out "$R/scripted_train.jsonl" || exit 1
step m_parity $PY scripts/parity.py configs/eval.yaml --samples "$P/val.jsonl" --n 20 run.out_dir="$R" \
  engine.max_model_len=$MAXLEN || exit 1
step m_cost_64k $PY scripts/cost_probe.py configs/eval.yaml --part engine --samples "$P/dev_64k.jsonl" --n 6 \
  run.out_dir="$R/cost_64k" engine.max_model_len=$MAXLEN || exit 1
step m_cost_128k $PY scripts/cost_probe.py configs/eval.yaml --part engine --samples "$P/dev_128k.jsonl" --n 3 \
  run.out_dir="$R/cost_128k" engine.max_model_len=$MAXLEN || exit 1
for B in sealedread nomask; do
  step "m_cost_sft_$B" $PY scripts/cost_probe.py configs/sft.yaml --part sft --steps 5 train.branch=$B \
    train.batch_traces=4 "data.samples=[$P/train.jsonl]" data.train="$R/scripted_train.jsonl" data.val=null \
    run.name=cost_sft_$B run.out_dir="$R/cost_sft_$B" || exit 1
done
# pilot S0 at 64K on train-shard questions (not the official gate, D26): (d) and (f) on 300 questions give
# EM(f) - EM(d) within about +-5 points; (a) 100 for context; (b) 150 for Pi; zero-shot <locate> recall on 100
S0=(run.name=s0 "run.out_dir=$R" "engine.max_model_len=$MAXLEN"
    "s0.dev_files={65536: $P/dev_64k.jsonl}"
    "s0.questions={d: 300, f: 300, a: 100, b: 150, c: 50, e: 50, at_32k: 20}")
step m_p0_vllm $PY scripts/p0_probe.py configs/s0.yaml --configs f,d,a "${S0[@]}" || exit 1
step m_p0_ref $PY scripts/p0_probe.py configs/s0.yaml --configs b "${S0[@]}" || exit 1
step m_recall $PY scripts/p0_probe.py configs/s0.yaml --recall "${S0[@]}" \
  "s0.recall_at_k={chunk_files: {256: $P/dev_64k.jsonl}, questions: 100}" || exit 1
$PY scripts/report.py p0 "$R/s0" --pilot 5 --out "$R/s0/p0_pilot.json" | tail -40
if [ "$GPU_GB" -ge 70 ]; then
  step m_teacher_pilot $PY scripts/teacher_traces.py configs/teacher.yaml --pilot engine.tensor_parallel_size=1 \
    engine.max_model_len=32768 engine.gpu_memory_utilization=0.92 teacher.samples="$P/train.jsonl" \
    teacher.reserve=null teacher.pilot=20 teacher.out="$R/teacher_pilot" || log "teacher pilot failed (see log)"
else
  log "skip teacher pilot: needs an 80 GB A100 for the 27B teacher"
fi
rsync -a "$R/" "$DRIVE/runs/measure/"
echo "MEASURE_DONE"
