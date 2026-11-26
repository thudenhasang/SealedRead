"""Overfit check: a run trained many epochs on a few traces must drive its loss close to 0. If it cannot, the
targets or the masks are wrong (a token the model is asked to predict is not visible / not consistent).

    python scripts/check_overfit.py /content/runs/measure/overfit_sealedread [--max-loss 0.3] [--max-ratio 0.2]
Pass: the mean loss of the last 3 logged steps is below --max-loss, or below --max-ratio x the first step's loss.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def verdict(losses: list[float], max_loss: float, max_ratio: float) -> dict:
    if len(losses) < 4:
        return {"pass": False, "reason": f"only {len(losses)} logged steps"}
    first, last = losses[0], sum(losses[-3:]) / 3
    ok = last <= max_loss or last <= max_ratio * first
    return {"pass": ok, "first": first, "last": last, "ratio": last / first if first else None, "steps": len(losses)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--max-loss", type=float, default=0.3)
    ap.add_argument("--max-ratio", type=float, default=0.2)
    args = ap.parse_args()
    path = Path(args.run_dir) / "metrics.jsonl"
    losses = [r["loss"] for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip()) if "loss" in r]
    v = verdict(losses, args.max_loss, args.max_ratio)
    (Path(args.run_dir) / "overfit_check.json").write_text(json.dumps(v, indent=1))
    print(json.dumps(v))
    print("OVERFIT_PASS" if v["pass"] else "OVERFIT_FAILED")
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
