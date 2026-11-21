"""S1a: teacher traces (section 5.3), with the quotas of table 4 and the trace filter.

    python scripts/teacher_traces.py configs/teacher.yaml --mode protocol      (SealedRead / no-mask / DA-ft traces)
    python scripts/teacher_traces.py configs/teacher.yaml --mode rag           (multi-round RAG: retriever locator)
    python scripts/teacher_traces.py configs/teacher.yaml --mode cot           (SEALONG: full-attention CoT)
    python scripts/teacher_traces.py configs/teacher.yaml --mode protocol --val 300     (validation traces)
    python scripts/teacher_traces.py configs/teacher.yaml --pilot              (S0: caps = p99 per tag, pass rates)
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from srd.cli import config_from_argv, log, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples, write_jsonl  # noqa: E402
from srd.engine.driver import DriverConfig, run_episodes  # noqa: E402
from srd.engine.factory import make_vllm_engine  # noqa: E402
from srd.engine.plain import run_plain  # noqa: E402
from srd.engine.retriever import make_retriever  # noqa: E402
from srd.engine.vllm_policy import VLLMPolicy  # noqa: E402
from srd.prompts import Prompts  # noqa: E402
from srd.protocol import MASKED_TAGS, ProtoConfig  # noqa: E402
from srd.reward import RewardConfig, r_ans  # noqa: E402
from srd.teacher import collect, teacher_ok  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def _flag(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def main() -> int:
    cfg, flags = config_from_argv()
    t = to_plain(cfg.teacher)
    out = Path(t["out"])
    mode = _flag(flags, "--mode", "protocol")
    ecfg = to_plain(cfg.engine)
    proto = ProtoConfig.from_cfg(to_plain(cfg.protocol))
    rcfg = RewardConfig.from_cfg(to_plain(cfg.reward))
    prompts = Prompts.from_cfg(to_plain(cfg.get("prompts")))
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    stok = load_tok(cfg.student_tokenizer)
    count = (lambda x: len(stok.encode(x)))
    engine = make_vllm_engine(ecfg, False)
    temp, top_p = ecfg.get("temperature", 0.7), ecfg.get("top_p", 1.0)
    val_n = _flag(flags, "--val")
    samples = load_samples(t["samples"] if not val_n else t["samples"].replace("train.jsonl", "val.jsonl"))
    if t.get("reserve") and not val_n:
        samples += load_samples(t["reserve"])
    if mode == "cot":
        recs = []
        outs = run_plain(engine, tok, prompts, samples[: 2 * t["total"]], "plain", temperature=temp, n=t["attempts"])
        by = {s.qid: s for s in samples}
        seen = set()
        for o in outs:
            if o["qid"] in seen or o["truncated"] or r_ans(o["answer"], by[o["qid"]]) < 1:
                continue
            seen.add(o["qid"])
            recs.append({"qid": o["qid"], "text": o["text"], "source": "teacher_cot"})
            if len(recs) >= t["total"]:
                break
        write_jsonl(out / "traces_cot.jsonl", recs)
        log(f"{len(recs)} CoT traces")
        print("TEACHER_DONE")
        return 0
    pol = VLLMPolicy(engine, tok, prompts, "masked" if mode == "rag" else "two_stream", proto, temp, top_p)
    drv = DriverConfig(proto=proto, locator="retriever" if mode == "rag" else "model")
    retr = make_retriever(to_plain(cfg.get("retriever"))) if mode == "rag" else None
    if "--pilot" in flags:
        ss = samples[: t["pilot"]]
        eps = run_episodes(pol, ss, 1, drv, retriever=retr)
        lens = defaultdict(list)
        why = defaultdict(int)
        for ep in eps:
            ok, w = teacher_ok(ep, proto, rcfg, None)
            why[w] += 1
            for st in ep.steps:
                if st.tag in MASKED_TAGS:
                    lens[st.tag].append(count(st.text))
        caps = {k: int(math.ceil(np.percentile(v, 99))) for k, v in lens.items() if v}
        write_json(out / "pilot.json", {"caps_p99": caps, "filter": dict(why), "n": len(eps)})
        log(f"pilot: caps (p99, student tokens) {caps}; filter {dict(why)}")
        print("TEACHER_PILOT_DONE")
        return 0
    total = int(val_n) if val_n else t["total"]
    recs, stats = collect(pol, samples, total, t["mix"], drv, rcfg, t["attempts"], t["batch"], count, retr,
                          log=log)
    name = {"protocol": "traces_protocol", "rag": "traces_rag"}[mode] + ("_val" if val_n else "")
    if val_n and mode == "protocol":
        name = "traces_val"
    write_jsonl(out / f"{name}.jsonl", recs)
    write_json(out / f"{name}.stats.json", stats)
    log(json.dumps(stats))
    print("TEACHER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
