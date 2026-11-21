"""S1 SFT / S2 M-step for one branch and seed.

    python scripts/train_sft.py configs/sft.yaml train.branch=sealedread run.name=sft_sr_s0 run.seed=0
    torchrun --nproc_per_node 8 scripts/train_sft.py configs/sft.yaml ...          (several GPUs)
    M-step: ... init_from=/content/runs/sft_sr_s0 data.train=/content/runs/rest_sr_s0/selected.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from srd.cli import config_from_argv, to_plain  # noqa: E402
from srd.train.sft import SFTTrainer  # noqa: E402


def main() -> int:
    cfg, _ = config_from_argv()
    out = Path(cfg.run.out_dir) / cfg.run.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(to_plain(cfg), sort_keys=False))
    SFTTrainer(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
