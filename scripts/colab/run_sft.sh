#!/usr/bin/env bash
# One SFT run within the session's time; rerun on the next runtime to resume from the last checkpoint.
#   bash scripts/colab/run_sft.sh configs/sft.yaml train.branch=nomask run.name=sft_nomask_s0 [overrides...]
source "$(dirname "$0")/common.sh"
CFG=$1
shift
NAME=$($PY -c "import sys; sys.path.insert(0, 'src'); from srd.config import load_config; \
print(load_config('$CFG', sys.argv[1:]).run.name)" "$@")
[ -f "$DRIVE/markers/train_$NAME.done" ] && { log "skip $NAME (done earlier)"; exit 0; }
[ -d "$DRIVE/runs/$NAME" ] && { mkdir -p "$RUNS/$NAME"; rsync -a "$DRIVE/runs/$NAME/" "$RUNS/$NAME/"; }
CAP=$(minutes_left 40)
log "train $NAME with $CAP minutes left in this session"
$PY scripts/train_sft.py "$CFG" "run.max_minutes=$CAP" "$@" 2>&1 | tee -a "$LOGS/train_$NAME.log"
rsync -a "$RUNS/$NAME/" "$DRIVE/runs/$NAME/"
if grep -q "TRAIN_DONE" "$LOGS/train_$NAME.log"; then
  touch "$DRIVE/markers/train_$NAME.done"; echo "TRAIN_PIPELINE_DONE $NAME"
else
  echo "TRAIN_PIPELINE_PAUSED $NAME (rerun on the next runtime to resume)"
fi
