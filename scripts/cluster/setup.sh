#!/usr/bin/env bash
# One-time environment on the GPU machine.  bash scripts/cluster/setup.sh
source "$(dirname "$0")/env.sh"
VLLM_SPEC=${VLLM_SPEC:-"vllm"}
command -v uv >/dev/null 2>&1 || pip install -q uv
[ -x "$WORK/venv/bin/python" ] || uv venv "$WORK/venv" --python 3.12
uv pip install --python "$WORK/venv/bin/python" "$VLLM_SPEC"
uv pip install --python "$WORK/venv/bin/python" -e "$REPO[train,data,rag,dev]"
$PY -m pytest -q "$REPO/tests" || { echo "tests fail on this machine"; exit 1; }
uv pip freeze --python "$WORK/venv/bin/python" > "$LOGS/venv.freeze.txt"
echo "SETUP_DONE"
