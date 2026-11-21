#!/usr/bin/env bash
# After a runtime reset: copy the built data and the runs back from Drive to the runtime's disk.
#   bash scripts/colab/restore.sh
source "$(dirname "$0")/common.sh"
command -v rsync >/dev/null 2>&1 || apt-get -qq install -y rsync >/dev/null 2>&1
[ -d "$DRIVE/data/srd" ] && mkdir -p "$DATA" && rsync -a "$DRIVE/data/srd/" "$DATA/"
[ -d "$DRIVE/runs" ] && mkdir -p "$RUNS" && rsync -a "$DRIVE/runs/" "$RUNS/"
echo "restored: $(du -sh "$DATA" 2>/dev/null | cut -f1) data, $(du -sh "$RUNS" 2>/dev/null | cut -f1) runs"
echo "RESTORE_DONE"
