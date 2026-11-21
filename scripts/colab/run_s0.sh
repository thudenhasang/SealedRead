#!/usr/bin/env bash
# S0 on one GPU: closed-book filter + check 2, paraphrases, teacher pilot (caps), engine parity (check 6), cost
# probe, the six P0 configs, the P0 decision. vLLM and the reference engine run in separate processes.
#   bash scripts/colab/run_s0.sh
source "$(dirname "$0")/common.sh"
step closed_book_heldout $PY scripts/closed_book.py configs/base.yaml --samples "$DATA/test_64k.jsonl" \
  "$DATA/test_fict_64k.jsonl" --out "$RUNS/closed_book/heldout.jsonl" || exit 1
step paraphrase $PY scripts/paraphrase.py configs/teacher.yaml "$DATA/train.jsonl" "$DATA/val.jsonl" "$DATA/reserve.jsonl" || exit 1
step check_after_paraphrase $PY scripts/check_data.py "$DATA"/train.jsonl "$DATA"/val.jsonl --limit 2000 \
  --closed-book "$RUNS/closed_book/heldout.jsonl" --out "$DATA/checks_after_paraphrase.json" || exit 1
step teacher_pilot $PY scripts/teacher_traces.py configs/teacher.yaml --pilot || exit 1
step parity $PY scripts/parity.py configs/eval.yaml --samples "$DATA/val.jsonl" --n 50 run.out_dir="$RUNS" || exit 1
step cost_engine $PY scripts/cost_probe.py configs/eval.yaml --part engine --samples "$DATA/dev_64k.jsonl" --n 20 \
  run.out_dir="$RUNS" || exit 1
step p0_vllm $PY scripts/p0_probe.py configs/s0.yaml --configs a,d,f || exit 1
step p0_ref $PY scripts/p0_probe.py configs/s0.yaml --configs b,c,e || exit 1
$PY scripts/report.py p0 "$RUNS/s0" --out "$RUNS/s0/p0_decision.json"
rsync -a --exclude 'adapters_live/' "$RUNS/" "$DRIVE/runs/"
echo "S0_DONE"
