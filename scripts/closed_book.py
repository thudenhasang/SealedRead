"""Closed-book answers of the base model (GPU, vLLM): the train filter (section 4.2) and data check 2.

    python scripts/closed_book.py configs/base.yaml --shards-dir /content/data/shards --shards 0,1,2,3,4,10,11,12,13,14 \
        --out /content/runs/closed_book/train.jsonl            # -> build.closed_book_filter
    python scripts/closed_book.py configs/base.yaml --samples /content/data/srd/test_64k.jsonl \
        /content/data/srd/test_fict_64k.jsonl --out /content/runs/closed_book/heldout.jsonl    # -> check 2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import config_from_argv, log, to_plain  # noqa: E402
from srd.data.sample import load_samples, write_jsonl  # noqa: E402
from srd.data.splits import load_items  # noqa: E402
from srd.engine.factory import make_vllm_engine  # noqa: E402
from srd.text import exact_match  # noqa: E402
from srd.tok import load_tok  # noqa: E402

PROMPT = ("<|im_start|>system\nAnswer from your own knowledge with a short phrase only.<|im_end|>\n"
          "<|im_start|>user\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n")


def _arg(flags, name, default=None):
    if name in flags:
        i = flags.index(name)
        vals = []
        for x in flags[i + 1:]:
            if x.startswith("--"):
                break
            vals.append(x)
        return vals
    return default


def main() -> int:
    cfg, flags = config_from_argv()
    out = _arg(flags, "--out")[0]
    rows = []
    if _arg(flags, "--samples"):
        for f in _arg(flags, "--samples"):
            for s in load_samples(f):
                rows.append({"qid": s.qid, "dataset": s.dataset, "qtype": s.qtype, "q": s.question, "a": s.answers})
    else:
        shards = [int(x) for x in _arg(flags, "--shards")[0].split(",")]
        for it in load_items(_arg(flags, "--shards-dir")[0], shards, ["hotpotqa", "2wikimultihopqa", "musique"]):
            rows.append({"qid": f"{it.dataset}-{it.qid}", "dataset": it.dataset, "qtype": "multihop",
                         "q": it.question, "a": it.answers})
    ecfg = to_plain(cfg.engine)
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    eng = make_vllm_engine({**ecfg, "max_model_len": 4096}, enable_lora=False)
    outs = eng.generate([tok.encode(PROMPT.format(q=r["q"])) for r in rows], max_tokens=32, temperature=0.0,
                        stop=["<|im_end|>"])
    recs = [{"qid": r["qid"], "dataset": r["dataset"], "qtype": r["qtype"], "answer": o.text.strip(),
             "correct": exact_match(o.text.strip(), r["a"])} for r, o in zip(rows, outs)]
    write_jsonl(out, recs)
    acc = sum(r["correct"] for r in recs) / max(1, len(recs))
    log(f"closed-book accuracy {acc:.3f} on {len(recs)} questions -> {out}")
    print("CLOSED_BOOK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
