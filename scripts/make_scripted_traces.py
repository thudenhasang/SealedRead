"""Protocol-valid traces from the scripted oracle policy (no model, CPU): for SFT cost probes and to top up the
rehearsal when a small teacher produces too few traces. NOT teacher data: never train the real models on them.

    python scripts/make_scripted_traces.py --samples data/train.jsonl --out runs/scripted.jsonl [--n 200] [--append]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.data.sample import load_samples  # noqa: E402
from srd.engine.driver import DriverConfig, run_episodes  # noqa: E402
from srd.engine.scripted import ScriptedPolicy  # noqa: E402
from srd.protocol import ProtoConfig, steps_to_dicts  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    have = set()
    if args.append and out.exists():
        have = {json.loads(x)["qid"] for x in open(out) if x.strip()}
    ss = [s for s in load_samples(args.samples) if s.qid not in have][: args.n]
    eps = run_episodes(ScriptedPolicy(noise=args.noise), ss, 1, DriverConfig())
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "a" if args.append else "w", encoding="utf-8") as f:
        for ep in eps:
            if ep.violations(ProtoConfig()):
                continue
            f.write(json.dumps({"qid": ep.sample.qid, "qtype": ep.sample.qtype, "steps": steps_to_dicts(ep.steps),
                                "source": "scripted"}) + "\n")
            n += 1
    print(f"{n} scripted traces -> {out}")
    print("SCRIPTED_TRACES_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
