#!/usr/bin/env bash
# SFT / M-step on all GPUs (torchrun; LoRA gradients all-reduced by the trainer).
#   bash scripts/cluster/train.sh configs/sft.yaml train.branch=sealedread run.name=sft_sr_s0 run.seed=0
source "$(dirname "$0")/env.sh"
NAME=$($PY -c "import sys; sys.path.insert(0, 'src'); from srd.config import load_config; \
print(load_config(sys.argv[1], sys.argv[2:]).run.name)" "$@")
$WORK/venv/bin/torchrun --standalone --nproc_per_node "$NGPU" scripts/train_sft.py "$@" $PATHS \
  2>&1 | tee -a "$LOGS/train_$NAME.log"
grep -q TRAIN_DONE "$LOGS/train_$NAME.log" && echo "TRAIN_PIPELINE_DONE $NAME"
