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
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1   # no usage reports sent out
export PATH=/content/venv-srd/bin:$PATH      # vLLM / torch.compile build kernels with the venv's ninja
export HF_HOME=${HF_HOME:-/content/hf}
mkdir -p "$LOGS" "$RUNS" "$DRIVE/markers" 2>/dev/null
cd "$REPO"
command -v rsync >/dev/null 2>&1 || apt-get -qq install -y rsync >/dev/null 2>&1

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

# ensure_shards: the raw multi-hop shards in /content/data/shards. From ragproj2's folder on Drive if it is there,
# else from a copy this project saved on Drive earlier, else rebuilt from Hugging Face (then saved on Drive).
# No marker: it checks the files themselves, so a reset runtime gets them back.
ensure_shards() {
  local out=/content/data/shards
  ls "$out"/*_s0.json >/dev/null 2>&1 && { log "shards present"; return 0; }
  mkdir -p "$out"
  if ls "$SHARDS"/*_s0.json >/dev/null 2>&1; then
    log "shards from Drive: $SHARDS"
    $PY scripts/fetch_data.py --from-drive "$SHARDS" --out "$out" > "$LOGS/fetch_shards.log" 2>&1
  elif ls "$DRIVE/data/shards"/*_s0.json >/dev/null 2>&1; then
    log "shards from this project's copy on Drive"
    cp "$DRIVE/data/shards"/*.json "$out"/
  else
    log "no shards on Drive: rebuilding them from Hugging Face (several minutes)"
    $PY scripts/fetch_data.py --from-hf --out "$out" --shards 5 > "$LOGS/fetch_shards.log" 2>&1 || {
      tail -20 "$LOGS/fetch_shards.log"; return 1; }
    mkdir -p "$DRIVE/data/shards" && cp "$out"/*.json "$DRIVE/data/shards"/
  fi
  ls "$out"/*_s0.json >/dev/null 2>&1 || { log "FAILED: no shard files in $out"; return 1; }
}

# save_data / restore_data <dir under /content/data>: keep a built data set on Drive (markers live on Drive, so the
# data they stand for must too)
save_data() { mkdir -p "$DRIVE/data/$(basename "$1")" && cp -a "$1/." "$DRIVE/data/$(basename "$1")/"; }
restore_data() {
  [ -f "$1/train.jsonl" ] && return 0
  if [ -d "$DRIVE/data/$(basename "$1")" ]; then
    mkdir -p "$1" && cp -a "$DRIVE/data/$(basename "$1")/." "$1/" && log "restored $(basename "$1") from Drive"
  fi
  return 0
}

# minutes left in this Colab session (24 h from the VM's boot), minus a reserve
minutes_left() {
  local reserve=${1:-30}
  local up=$(awk '{print int($1/60)}' /proc/uptime)
  echo $(( 24 * 60 - up - reserve ))
}
