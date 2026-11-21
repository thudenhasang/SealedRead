"""Data checks 1, 3, 4, 5 (+ structure, scripted oracle run; 2 from a closed-book file) with the pre-registered
thresholds. Exit code 1 if a check fails.

    python scripts/check_data.py /content/data/srd/*.jsonl --out /content/data/srd/checks.json \
        [--closed-book /content/runs/closed_book/heldout.jsonl] [--limit 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from srd.cli import write_json  # noqa: E402
from srd.data import checks  # noqa: E402
from srd.data.sample import load_samples, read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="samples per file (the checks are per-sample)")
    ap.add_argument("--closed-book", default=None)
    ap.add_argument("--prereg", default="configs/prereg.yaml")
    args = ap.parse_args()
    pre = yaml.safe_load(open(args.prereg))["data_checks"]
    report, ok = {}, True
    for f in args.files:
        if not f.endswith(".jsonl"):
            continue
        ss = load_samples(f, args.limit)
        name = Path(f).name
        r = {"structure": checks.check_structure(ss),
             "gold_mapping": checks.check_gold_mapping(ss, pre["gold_mapping_min"]),
             "leaks": checks.check_leaks(ss, pre["leak_clean_min"]),
             "surface": checks.check_surface(ss, pre["surface_auc_max"], title_ngrams="fict" in name),
             "paraphrase": checks.check_paraphrase(ss, pre["paraphrase_keep_min"]),
             "scripted": checks.check_scripted(ss[:200])}
        report[name] = r
        bad = [k for k, v in r.items() if not v["pass"]]
        ok &= not bad
        print(f"{name}: {len(ss)} samples, " + ("all checks pass" if not bad else f"FAILED {bad}"), flush=True)
    if args.closed_book:
        r = checks.check_closed_book(list(read_jsonl(args.closed_book)), pre["closed_book_max_multihop"],
                                     pre["closed_book_max_fictional"])
        report["closed_book"] = r
        print(f"closed-book: {r['acc']} -> {'pass' if r['pass'] else 'FAILED'} {r.get('fallback') or ''}")
        ok &= r["pass"]
    if args.out:
        write_json(args.out, report)
    print("DATA_CHECKS_PASS" if ok else "DATA_CHECKS_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
