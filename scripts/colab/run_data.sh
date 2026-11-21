#!/usr/bin/env bash
# M1 on Colab: shards from Drive -> tokenizer check -> build (exact token budgets) -> data checks -> Drive.
# Paraphrasing and the closed-book filter need the GPU: run_s0.sh does them. Resumable through Drive markers.
#   bash scripts/colab/run_data.sh [extra build overrides...]
source "$(dirname "$0")/common.sh"
OUT=${OUT:-$DATA}
step fetch_shards $PY scripts/fetch_data.py --from-drive "$SHARDS" --out /content/data/shards || exit 1
step tokenizer_check $PY scripts/tokenizer_check.py Qwen/Qwen3-4B-Instruct-2507 --out "$RUNS/tokenizer_check.json" || exit 1
step "build_$(basename "$OUT")" $PY scripts/build_data.py --config configs/data.yaml --shards-dir /content/data/shards \
  --out "$OUT" "$@" || exit 1
step "check_$(basename "$OUT")" $PY scripts/check_data.py "$OUT"/*.jsonl --limit 300 --out "$OUT/checks.json" || exit 1
mkdir -p "$DRIVE/data/$(basename "$OUT")"
rsync -a "$OUT/" "$DRIVE/data/$(basename "$OUT")/"
echo "DATA_DONE"
