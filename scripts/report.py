"""Summaries and the pre-registered decisions from result files (CPU).

    python scripts/report.py summary <records.jsonl>...                       aggregate metrics per file
    python scripts/report.py p0 <s0 dir>                                      Delta_oracle, Pi, table 6 action
    python scripts/report.py s1 --sr <files> --nm <files>                     decision point after S1 (dev 64K)
    python scripts/report.py s2 --sr <files> --a <files>                      gate before S2 (validity, EM vs (a))
    python scripts/report.py rq1 --sr <seed files...> --nm <seed files...> [--sr32 ... --nm32 ...]
    python scripts/report.py rq2 --sr <files> --rag <files>
    python scripts/report.py round2 --after <files> --before <files>
    python scripts/report.py compare --a <files> --b <files> [--key em]       any paired difference (bootstrap)
Seed files: one file per seed, separated by commas: --sr s0.jsonl,s1.jsonl,s2.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from srd.eval import gates  # noqa: E402
from srd.eval.metrics import aggregate, by_question  # noqa: E402
from srd.eval.runner import load_records  # noqa: E402
from srd.eval.stats import paired_bootstrap, scale  # noqa: E402


def recs(files: list[str]) -> list[dict]:
    out = []
    for f in files:
        for g in sorted(glob.glob(f)) or [f]:
            out += load_records(g)
    return out


def q(files: list[str], key: str = "em") -> dict:
    return by_question(recs(files), key)


def seeds(arg: str | None) -> list[dict]:
    return [q([f]) for f in arg.split(",")] if arg else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("files", nargs="*")
    for k in ("sr", "nm", "a", "b", "rag", "after", "before", "sr32", "nm32"):
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--key", default="em")
    ap.add_argument("--prereg", default="configs/prereg.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pre = yaml.safe_load(open(args.prereg))
    split = (lambda s: s.split(",") if s else [])
    if args.cmd == "summary":
        res = {Path(f).name: aggregate(load_records(f)) for f in args.files}
    elif args.cmd == "p0":
        d = Path(args.files[0])
        ls = pre["p0"]["lengths"]
        get = (lambda c: q([str(d / f"{c}_{L // 1024}k*.jsonl") for L in ls]))
        res = gates.p0_decision(get("b"), get("d"), get("f"), pre)
        res["other"] = {}
        for x, y, name in (("b", "c", "sandwich_gain"), ("b", "a", "b_minus_a"), ("e", "b", "e_minus_b")):
            try:
                res["other"][name] = scale(paired_bootstrap(get(x), get(y)))
            except ValueError:
                pass
    elif args.cmd == "s1":
        res = gates.s1_decision(q(split(args.sr)), q(split(args.nm)), pre)
    elif args.cmd == "s2":
        rs = recs(split(args.sr))
        valid = sum(bool(r.get("valid")) for r in rs) / max(1, len(rs))
        res = gates.s2_gate(valid, by_question(rs), q(split(args.a)), pre)
    elif args.cmd == "rq1":
        res = gates.rq1(seeds(args.sr), seeds(args.nm), seeds(args.sr32), seeds(args.nm32), pre)
    elif args.cmd == "rq2":
        res = gates.rq2(q(split(args.sr), args.key), q(split(args.rag), args.key), pre)
    elif args.cmd == "round2":
        res = gates.round2(q(split(args.after)), q(split(args.before)), pre)
    elif args.cmd == "compare":
        res = scale(paired_bootstrap(q(split(args.a), args.key), q(split(args.b), args.key)))
    else:
        raise SystemExit(f"unknown command {args.cmd}")
    text = json.dumps(res, indent=1, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
