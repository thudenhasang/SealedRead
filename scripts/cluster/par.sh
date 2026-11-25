#!/usr/bin/env bash
# Run one python script on all GPUs at once, each process on its own share of the questions
# (SHARD / NUM_SHARDS, read by the scripts), then wait for all.
#   bash scripts/cluster/par.sh <name> scripts/evaluate.py configs/eval.yaml eval.branch=sealedread ...
#   GPUS_PER_PROC=2 bash scripts/cluster/par.sh teacher scripts/teacher_traces.py configs/teacher.yaml ...
#     (tensor parallel 2: 4 processes on 8 GPUs; engine.tensor_parallel_size must equal GPUS_PER_PROC)
source "$(dirname "$0")/env.sh"
NAME=$1
shift
G=${GPUS_PER_PROC:-1}
N=${NUM_SHARDS:-$((NGPU / G))}
pids=()
for ((i = 0; i < N; i++)); do
  DEVS=$(seq -s, $((i * G)) $((i * G + G - 1)))
  CUDA_VISIBLE_DEVICES=$DEVS SHARD=$i NUM_SHARDS=$N $PY "$@" $PATHS > "$LOGS/${NAME}_$i.log" 2>&1 &
  pids+=($!)
  sleep 5                                   # stagger the engine start-ups (ports, model loading)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] && echo "PAR_DONE $NAME" || { echo "PAR_FAILED $NAME (see $LOGS/${NAME}_*.log)"; exit 1; }
