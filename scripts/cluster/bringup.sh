#!/usr/bin/env bash
# Day-1 bring-up on the GPU machine (D27), about 1-2 hours, after setup.sh and before the official S0. It checks
# what one Colab GPU could not: this machine's kernels (parity), its real speed (costs), and several GPUs at once
# (par.sh shards, torchrun, 4 teacher processes x tensor parallel 2 + merge). Every item prints PASS / FAIL, then a
# summary; the exit code is 1 if any item failed. Logs: $LOGS/bringup_<item>.log, results: $RUNS/bringup.
#   bash scripts/cluster/bringup.sh
source "$(dirname "$0")/env.sh"
B=$RUNS/bringup
BD=$WORK/data/bringup
M=Qwen/Qwen3-4B-Instruct-2507
mkdir -p "$B" "$BD"
ITEMS=()
declare -A RES

check() {   # check <item> <command...>
  local item=$1
  shift
  ITEMS+=("$item")
  log "== $item"
  if "$@" > "$LOGS/bringup_$item.log" 2>&1; then
    RES[$item]=PASS
  else
    RES[$item]=FAIL
    tail -15 "$LOGS/bringup_$item.log"
  fi
  log "$item: ${RES[$item]}"
}

log "GPUs: $NGPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -8

# --- one GPU: code, data, kernels, speed
check tests $PY -m pytest -q tests
check tokenizer $PY scripts/tokenizer_check.py $M --out "$B/tokenizer_check.json"
ls "$SHARDS"/*_s0.json > /dev/null 2>&1 || \
  check shards $PY scripts/fetch_data.py --from-hf --out "$SHARDS" --shards 5
check data $PY scripts/build_data.py --config configs/data.yaml --shards-dir "$SHARDS" --out "$BD" \
  "build.train_shards=[0]" build.eval_shard=4 build.train.n=24 build.val.n=10 build.reserve.n=10 build.dev.n=10 \
  "build.dev.lengths={65536: 10, 131072: 5}" build.test.n=10 "build.test.lengths={65536: 10}" \
  "build.test.fictional_lengths=[]"
check traces $PY scripts/make_scripted_traces.py --samples "$BD/train.jsonl" --out "$B/scripted.jsonl"
check parity env CUDA_VISIBLE_DEVICES=0 $PY scripts/parity.py configs/eval.yaml --samples "$BD/val.jsonl" --n 10 \
  run.out_dir="$B"
check cost_64k env CUDA_VISIBLE_DEVICES=0 $PY scripts/cost_probe.py configs/eval.yaml --part engine \
  --samples "$BD/dev_64k.jsonl" --n 6 run.out_dir="$B/cost_64k"
check cost_128k env CUDA_VISIBLE_DEVICES=0 $PY scripts/cost_probe.py configs/eval.yaml --part engine \
  --samples "$BD/dev_128k.jsonl" --n 3 run.out_dir="$B/cost_128k"
check cost_sft env CUDA_VISIBLE_DEVICES=0 $PY scripts/cost_probe.py configs/sft.yaml --part sft --steps 5 \
  train.branch=sealedread train.batch_traces=4 "data.samples=[$BD/train.jsonl]" data.train="$B/scripted.jsonl" \
  data.val=null run.name=cost_sft run.out_dir="$B/cost_sft"

# --- several GPUs
check par_eval env NUM_SHARDS=2 bash scripts/cluster/par.sh bringup_eval scripts/evaluate.py configs/eval.yaml \
  eval.branch=sealedread eval.lenient=true "eval.files=[$BD/dev_64k.jsonl]" eval.limit=10 eval.out="$B/eval"
check par_eval_rows test "$(cat "$B"/eval/sealedread__dev_64k.shard*of2.jsonl 2>/dev/null | wc -l)" -eq 10
check torchrun_sft bash scripts/cluster/train.sh configs/sft.yaml train.branch=sealedread \
  "data.samples=[$BD/train.jsonl]" data.train="$B/scripted.jsonl" data.limit=24 data.val=null train.epochs=1 \
  train.batch_traces=8 run.name=bringup_sft run.out_dir="$B"
check teacher_par env GPUS_PER_PROC=2 bash scripts/cluster/par.sh bringup_teacher scripts/teacher_traces.py \
  configs/teacher.yaml --mode protocol teacher.samples="$BD/train.jsonl" teacher.reserve=null teacher.total=8 \
  teacher.out="$B/teacher"
check teacher_merge $PY scripts/teacher_traces.py configs/teacher.yaml --mode protocol --merge teacher.total=8 \
  teacher.out="$B/teacher"

echo
echo "bring-up summary"
fail=0
for item in "${ITEMS[@]}"; do
  printf "  %-16s %s\n" "$item" "${RES[$item]}"
  [ "${RES[$item]}" = PASS ] || fail=1
done
echo "results: $B (parity.json, cost_*/, teacher/), logs: $LOGS/bringup_*.log"
[ $fail = 0 ] && echo "BRINGUP_PASS: go on with  bash scripts/cluster/pipeline.sh s0" || { echo "BRINGUP_FAILED"; exit 1; }
