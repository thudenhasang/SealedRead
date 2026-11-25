"""S0 (P0): the six zero-shot configurations and the recall of a zero-shot <locate> (section 5.2).

    (a) full attention, question after C                         vLLM, plain prompt
    (b) oracle paste + C masked, sandwich layout (main design)   reference engine, masked view, full frame (no C prefill)
    (c) as (b) with only Q before C (no Q')                      reference engine
    (d) oracle paste, C not masked, sandwich                     vLLM, nomask
    (e) oracle original KV of the segments of G, no paste        reference engine, daft chain (C prefilled)
    (f) as (b) with C removed and positions compacted            vLLM, masked mode (compact frame)
Main design = two streams, compact frame: the gate compares (f) with (d); (b) only measures Pi (positional share);
(c) and (e) are optional. vLLM and the reference engine do not share a GPU: run them in two processes.

    python scripts/p0_probe.py configs/s0.yaml --configs a,d,f
    python scripts/p0_probe.py configs/s0.yaml --configs b
    python scripts/p0_probe.py configs/s0.yaml --recall          (zero-shot <locate> Recall@K per chunk size)
    python scripts/report.py p0 /content/runs/s0                  (gate (f) vs (d), and Pi)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import config_from_argv, log, shard_args, shard_suffix, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples  # noqa: E402
from srd.engine.driver import DriverConfig, Episode  # noqa: E402
from srd.engine.factory import load_ref_model, make_vllm_engine  # noqa: E402
from srd.engine.plain import run_plain  # noqa: E402
from srd.eval.runner import evaluate, shard  # noqa: E402
from srd.prompts import Prompts  # noqa: E402
from srd.protocol import ProtoConfig, Step  # noqa: E402
from srd.reward import RewardConfig, precision_recall, r_ans  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def plain_records(samples, outs) -> list[dict]:
    by = {s.qid: s for s in samples}
    recs = []
    for o in outs:
        s = by[o["qid"]]
        recs.append({"qid": s.qid, "cluster": s.cluster, "length": s.length, "qtype": s.qtype, "j": o["j"],
                     "answer_kind": s.answer_kind, "n_gold": len(s.gold), "answer": o["answer"],
                     "em": r_ans(o["answer"], s), "valid": not o["truncated"]})
    return recs


def main() -> int:
    cfg, flags = config_from_argv()
    s0 = to_plain(cfg.s0)
    out = Path(cfg.run.out_dir) / cfg.run.name
    proto = ProtoConfig.from_cfg(to_plain(cfg.protocol))
    rcfg = RewardConfig.from_cfg(to_plain(cfg.reward))
    prompts = Prompts.from_cfg(to_plain(cfg.get("prompts")))
    ecfg = to_plain(cfg.engine)
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    i, n = shard_args()
    if "--recall" in flags:
        from srd.engine.vllm_policy import VLLMPolicy

        pol = VLLMPolicy(make_vllm_engine(ecfg, False), tok, prompts, "two_stream", proto)
        res = {}
        for size, f in s0["recall_at_k"]["chunk_files"].items():
            if not f:
                continue
            ss = [s for s in load_samples(f, s0["recall_at_k"]["questions"]) if s.gold]
            eps = [Episode(s, [Step("reason", "I need the evidence. alpha=query"), Step("query", s.question)])
                   for s in ss]
            got = pol.gen_locate(eps, [[g.sid for g in s.segments] for s in ss])
            rec = [precision_recall(s.gold, ids)[1] for s, ids in zip(ss, got)]
            res[str(size)] = {"recall_at_k": sum(rec) / max(1, len(rec)), "n": len(rec)}
            log(f"chunk {size}: Recall@{proto.k} = {res[str(size)]['recall_at_k']:.3f}")
        write_json(out / "recall_at_k.json", res)
        print("P0_RECALL_DONE")
        return 0
    configs = (flags[flags.index("--configs") + 1] if "--configs" in flags else ",".join(s0["configs"])).split(",")
    engine = model = None
    for c in configs:
        backend = s0["backend"][c]
        if backend == "vllm" and engine is None:
            engine = make_vllm_engine(ecfg, False)
        if backend == "ref" and model is None:
            model = load_ref_model(ecfg)
        lengths = [int(x) for x in s0["dev_files"]]
        for length in lengths:
            nq = s0["questions"]["at_32k"] if length <= 32768 else s0["questions"][c]
            samples = shard(load_samples(s0["dev_files"][length], nq), i, n)
            path = shard_suffix(out / f"{c}_{length // 1024}k.jsonl")
            log(f"config ({c}) at {length // 1024}K: {len(samples)} questions")
            if c == "a":
                recs = plain_records(samples, run_plain(engine, tok, prompts, samples, "plain"))
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
                continue
            drv = DriverConfig(proto=proto, locator="oracle", lenient=s0.get("lenient", True))
            if c in ("b", "c"):
                from srd.engine.ref import RefPolicy

                pol = RefPolicy(model, tok, prompts, view="masked", frame="full", repeat_q=c == "b", proto=proto)
            elif c == "e":
                from srd.engine.ref import RefPolicy

                pol = RefPolicy(model, tok, prompts, branch="daft", view="chain", proto=proto)
            else:
                from srd.engine.vllm_policy import VLLMPolicy

                pol = VLLMPolicy(engine, tok, prompts, "nomask" if c == "d" else "masked", proto)
            evaluate(pol, samples, 1, drv, rcfg, path, prompts, extra={"config": c}, log=log,
                     batch=1 if backend == "ref" else 128)      # reference engine: one KV cache at a time
    print("P0_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
