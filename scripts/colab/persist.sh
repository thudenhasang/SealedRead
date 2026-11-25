#!/usr/bin/env bash
# Copy runs and logs from the runtime's disk to Drive every few minutes (and once more on exit), so a reset loses at
# most one interval. Start it in the background once per runtime:
#   nohup bash scripts/colab/persist.sh 300 > /content/logs/persist.log 2>&1 &
source "$(dirname "$0")/common.sh"
EVERY=${1:-300}
command -v rsync >/dev/null 2>&1 || apt-get -qq install -y rsync >/dev/null 2>&1
mkdir -p "$DRIVE/runs" "$DRIVE/logs"

sync_once() {
  # live adapters are rewritten every step and are re-made from the checkpoints: not copied
  [ -d "$RUNS" ] && rsync -a --exclude 'adapters_live/' "$RUNS/" "$DRIVE/runs/"
  [ -d "$LOGS" ] && rsync -a "$LOGS/" "$DRIVE/logs/"
  [ -d /content/data ] && rsync -a --exclude 'shards/' /content/data/ "$DRIVE/data/"
  date +%H:%M:%S > "$DRIVE/last_persist.txt"
}

trap sync_once EXIT
while true; do
  sync_once
  sleep "$EVERY"
done
