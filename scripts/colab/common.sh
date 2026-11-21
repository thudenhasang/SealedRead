#!/usr/bin/env bash
# Shared settings and helpers for the Colab pipelines (from keychain-dv). Source it: `source scripts/colab/common.sh`.
#   DRIVE    the project's folder on Google Drive (data, runs, logs, markers)
#   SHARDS   ragproj2's raw shards on Drive (read only): the same files keychain-dv used
#   PY       the environment made by setup.sh
set -uo pipefail
export REPO=${REPO:-/content/sealedread}
export DRIVE=${DRIVE:-/content/drive/MyDrive/sealedread}
export SHARDS=${SHARDS:-/content/drive/MyDrive/ragproj2/data/train}
export PY=${PY:-/content/venv-srd/bin/python}
export DATA=${DATA:-/content/data/srd}
export RUNS=${RUNS:-/content/runs}
export LOGS=${LOGS:-/content/logs}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/content/venv-srd/bin:$PATH      # vLLM / torch.compile build kernels with the venv's ninja
export HF_HOME=${HF_HOME:-/content/hf}
mkdir -p "$LOGS" "$RUNS" "$DRIVE/markers" 2>/dev/null
cd "$REPO"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# step <marker> <command...>: run once; a marker on Drive makes a rerun skip it (resumable pipelines)
step() {
  [ -f "$DRIVE/markers/$1.done" ] && { log "skip $1 (done earlier)"; return 0; }
  log "start $1"
  if "${@:2}" > "$LOGS/step_$1.log" 2>&1; then
    touch "$DRIVE/markers/$1.done"; log "done $1"
  else
    log "FAILED $1"; tail -20 "$LOGS/step_$1.log"; return 1
  fi
}

# minutes left in this Colab session (24 h from the VM's boot), minus a reserve
minutes_left() {
  local reserve=${1:-30}
  local up=$(awk '{print int($1/60)}' /proc/uptime)
  echo $(( 24 * 60 - up - reserve ))
}
