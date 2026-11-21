#!/usr/bin/env bash
# Run one python script on every GPU at once, each process on its own share of the questions
# (SHARD / NUM_SHARDS, read by the scripts), then wait for all. For engines that fit on one GPU (4B).
#   bash scripts/cluster/par.sh <name> scripts/evaluate.py configs/eval.yaml eval.branch=sealedread ...
source "$(dirname "$0")/env.sh"
NAME=$1
shift
N=${NUM_SHARDS:-$NGPU}
pids=()
for ((i = 0; i < N; i++)); do
  CUDA_VISIBLE_DEVICES=$i SHARD=$i NUM_SHARDS=$N $PY "$@" $PATHS > "$LOGS/${NAME}_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] && echo "PAR_DONE $NAME" || { echo "PAR_FAILED $NAME (see $LOGS/${NAME}_*.log)"; exit 1; }
