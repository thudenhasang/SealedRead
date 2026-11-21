#!/usr/bin/env bash
# Paths on the GPU machine (8 x B200). Source it: `source scripts/cluster/env.sh`. Override with environment vars.
set -uo pipefail
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export WORK=${WORK:-/data/sealedread}           # fast local disk
export DATA=${DATA:-$WORK/data/srd}
export RUNS=${RUNS:-$WORK/runs}
export LOGS=${LOGS:-$WORK/logs}
export SHARDS=${SHARDS:-$WORK/shards}
export PY=${PY:-$WORK/venv/bin/python}
export NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
export HF_HOME=${HF_HOME:-$WORK/hf}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$WORK/venv/bin:$PATH
mkdir -p "$DATA" "$RUNS" "$LOGS"
cd "$REPO"
# configs read ${DATA} and ${RUNS} (with Colab defaults), so nothing else needs overriding
export PATHS=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
