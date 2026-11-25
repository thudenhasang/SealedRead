"""S5 evaluation: one branch (or baseline) over sample files; one record per episode (resumable, sharded).

    python scripts/evaluate.py configs/eval.yaml eval.branch=sealedread engine.adapter=/content/runs/rest_sr_s0/best/adapter
    python scripts/evaluate.py configs/eval.yaml eval.branch=nomask engine.adapter=...
    python scripts/evaluate.py configs/eval.yaml eval.branch=plain            (full attention zero-shot, question after C)
    python scripts/evaluate.py configs/eval.yaml eval.branch=copy             (copy-then-solve prompt)
    python scripts/evaluate.py configs/eval.yaml eval.branch=rag1             (one-pass RAG)
    python scripts/evaluate.py configs/eval.yaml eval.branch=rag engine.adapter=...  (multi-round RAG, trained)
    python scripts/evaluate.py configs/eval.yaml eval.branch=daft engine.backend=ref engine.adapter=...
    RQ4 / ablations: eval.permute_ids=true | eval.oracle=true | eval.repeat_q=false | eval.locate_query_only=true |
                     prompts.show_titles=false | eval.hide_locate=false engine.backend=ref
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import config_from_argv, log, shard_args, shard_suffix, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples  # noqa: E402
from srd.engine.driver import DriverConfig  # noqa: E402
from srd.engine.factory import make_policy, make_vllm_engine  # noqa: E402
from srd.engine.plain import run_plain, run_rag_one_pass  # noqa: E402
from srd.engine.retriever import make_retriever  # noqa: E402
from srd.eval.metrics import aggregate  # noqa: E402
from srd.eval.runner import evaluate, load_records, shard  # noqa: E402
from srd.layout import MaskOpts  # noqa: E402
from srd.prompts import Prompts  # noqa: E402
from srd.protocol import ProtoConfig  # noqa: E402
from srd.reward import RewardConfig, answer_f1, r_ans  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def main() -> int:
    cfg, _ = config_from_argv()
    e = to_plain(cfg.eval)
    ecfg = to_plain(cfg.engine)
    proto = ProtoConfig.from_cfg(to_plain(cfg.protocol))
    rcfg = RewardConfig.from_cfg(to_plain(cfg.reward))
    prompts = Prompts.from_cfg(to_plain(cfg.get("prompts")))
    out = Path(e["out"])
    i, n = shard_args()
    tag = e.get("tag") or e["branch"]
    branch = e["branch"]
    pol = tok = retr = None
    if branch in ("plain", "copy", "rag1"):
        tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
        engine = make_vllm_engine(ecfg, bool(ecfg.get("adapter")))
        engine.set_adapter(ecfg.get("adapter"))
        if branch == "rag1":
            retr = make_retriever(to_plain(cfg.get("retriever")))
    else:
        opts = None
        if ecfg.get("backend") == "ref":
            mb = {"nomask": "nomask", "daft": "daft"}.get(branch, "sealedread")
            opts = MaskOpts(branch=mb, hide_locate=e.get("hide_locate", True),
                            locate_history="query_only" if e.get("locate_query_only") else "all")
        pol, tok, locator = make_policy(ecfg, branch, proto, prompts, opts=opts, repeat_q=e.get("repeat_q", True),
                                        locate_query_only=e.get("locate_query_only", False))
        if locator == "retriever":
            retr = make_retriever(to_plain(cfg.get("retriever")))
        drv = DriverConfig(proto=proto, locator="oracle" if e.get("oracle") else locator,
                           lenient=e.get("lenient", False), permute_ids=e.get("permute_ids", False))
    count = (lambda t: len(tok.encode(t)))
    for f in e["files"]:
        samples = shard(load_samples(f, e.get("limit")), i, n)
        path = shard_suffix(out / f"{tag}__{Path(f).stem}.jsonl")
        log(f"{tag} on {Path(f).name}: {len(samples)} questions")
        if pol is None:
            outs = (run_rag_one_pass(engine, tok, prompts, samples, retr, proto.k) if branch == "rag1"
                    else run_plain(engine, tok, prompts, samples, branch))
            by = {s.qid: s for s in samples}
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                for o in outs:
                    s = by[o["qid"]]
                    fh.write(json.dumps({"qid": s.qid, "cluster": s.cluster, "length": s.length, "qtype": s.qtype,
                                         "j": 0, "answer_kind": s.answer_kind, "n_gold": len(s.gold),
                                         "answer": o["answer"], "em": r_ans(o["answer"], s),
                                         "f1": answer_f1(o["answer"], s), "valid": not o["truncated"]}) + "\n")
            continue
        # the reference engine keeps one KV cache (with all of C) per live episode: one question at a time
        batch = 1 if ecfg.get("backend") == "ref" else 128
        evaluate(pol, samples, e.get("n", 1), drv, rcfg, path, prompts, retriever=retr, token_count=count, batch=batch,
                 extra={"branch": tag}, log=log)
        if n == 1 and pol is not None:
            agg = aggregate(load_records(path))
            write_json(path.with_suffix(".summary.json"), agg)
            log(f"{path.name}: " + " ".join(f"{k}={v:.3f}" for k, v in agg.items() if isinstance(v, float)))
    print("EVAL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
