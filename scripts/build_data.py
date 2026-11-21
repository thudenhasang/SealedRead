"""M1: build every sample file (section 4) from the multi-hop shards.

    python scripts/build_data.py --config configs/data.yaml --shards-dir /content/data/shards --out /content/data/srd
    [build.train.n=300 ...]   (dotted overrides of the config)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import log, token_counter, to_plain, write_json  # noqa: E402
from srd.config import load_config  # noqa: E402
from srd.data.splits import build_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--shards-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    b = to_plain(cfg.build)
    t0 = time.time()
    count = token_counter(b.get("tokenizer"))
    man = build_all(b, args.shards_dir, args.out, count, args.seed, log)
    man.update(config=b, minutes=round((time.time() - t0) / 60, 1))
    write_json(Path(args.out) / "manifest.json", man)
    log(f"files: {man['files']}")
    print("BUILD_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
