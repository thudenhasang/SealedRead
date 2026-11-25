#!/usr/bin/env bash
# Create the environment on a fresh Colab runtime (a few minutes): vLLM first (it pins torch), then this repo.
#   VLLM_SPEC   the vLLM version; pin the one the cost probe validated (docs/results/)
#   bash scripts/colab/setup.sh
set -euo pipefail
REPO=${REPO:-/content/sealedread}
VENV=${VENV:-/content/venv-srd}
VLLM_SPEC=${VLLM_SPEC:-"vllm"}
mkdir -p /content/logs
command -v uv >/dev/null 2>&1 || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12
uv pip install --python "$VENV/bin/python" "$VLLM_SPEC"
uv pip install --python "$VENV/bin/python" -e "$REPO[train,data,rag,dev]"
"$VENV/bin/python" - <<'PYEOF'
import torch, transformers, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "| vllm", vllm.__version__,
      "| transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PYEOF
uv pip freeze --python "$VENV/bin/python" > /content/logs/venv-srd.freeze.txt
echo "SETUP_DONE"
