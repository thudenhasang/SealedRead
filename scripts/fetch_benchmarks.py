"""RQ2 benchmarks -> samples at 32K / 64K / 128K (QASPER, NarrativeQA, QuALITY), via Hugging Face datasets.

    python scripts/fetch_benchmarks.py --config configs/benchmarks.yaml [--limit 50]
Prints one converted sample per benchmark: check the field mapping on the first run (dataset copies differ).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import log, token_counter, to_plain  # noqa: E402
from srd.config import load_config  # noqa: E402
from srd.data import benchmarks as B  # noqa: E402
from srd.data.sample import write_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/benchmarks.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = to_plain(load_config(args.config, args.overrides).benchmarks)
    from datasets import load_dataset

    count = token_counter(cfg.get("tokenizer"))
    out = Path(cfg["out"])
    for name, conv, kind in (("qasper", B.qasper_docs, "paper"), ("narrativeqa", None, "story"),
                             ("quality", B.quality_docs, "article")):
        c = cfg[name]
        ds = load_dataset(c["hf"], split=c["split"])
        if name == "narrativeqa":
            docs = B.narrativeqa_docs(ds, c.get("max_words", 90000))
        else:
            docs = conv(ds)
        if args.limit:
            docs = docs[: args.limit]
        lengths = cfg["lengths"] if c.get("padded", True) else [max(cfg["lengths"])]
        ss = B.build_docs_samples(docs, name, lengths, count, limit=c.get("limit"), kind=kind)
        for L in lengths:
            rows = [s.to_dict() for s in ss if s.length == L]
            write_jsonl(out / f"{name}_{L // 1024}k.jsonl", rows)
            log(f"{name} {L // 1024}K: {len(rows)} samples")
        if ss:
            s = ss[0]
            log(f"example {name}: Q={s.question[:120]!r} A={s.answers[:2]} segments={len(s.segments)} gold={len(s.gold)}")
    print("BENCHMARKS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
