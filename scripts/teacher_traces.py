"""S1a: teacher traces (section 5.3), with the quotas of table 4 and the trace filter.

    python scripts/teacher_traces.py configs/teacher.yaml --mode protocol      (SealedRead / no-mask / DA-ft traces)
    python scripts/teacher_traces.py configs/teacher.yaml --mode rag           (multi-round RAG: retriever locator)
    python scripts/teacher_traces.py configs/teacher.yaml --mode cot           (SEALONG: full-attention CoT)
    python scripts/teacher_traces.py configs/teacher.yaml --mode protocol --val 300     (validation traces)
    python scripts/teacher_traces.py configs/teacher.yaml --pilot              (S0: caps = p99 per tag, pass rates)

Several teacher processes at once (8 GPUs, tensor parallel 2 -> 4 processes; scripts/cluster/par.sh sets SHARD and
NUM_SHARDS): each takes its share of the questions and of every type's quota and writes <name>.shardIofN.jsonl;
then one call with --merge joins the shares into <name>.jsonl, within the global quotas:
    GPUS_PER_PROC=2 bash scripts/cluster/par.sh teacher scripts/teacher_traces.py configs/teacher.yaml --mode protocol
    python scripts/teacher_traces.py configs/teacher.yaml --mode protocol --merge
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from srd.cli import config_from_argv, log, shard_args, shard_suffix, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples, read_jsonl, write_jsonl  # noqa: E402
from srd.engine.driver import DriverConfig, run_episodes  # noqa: E402
from srd.engine.factory import make_vllm_engine  # noqa: E402
from srd.engine.plain import run_plain  # noqa: E402
from srd.engine.retriever import make_retriever  # noqa: E402
from srd.engine.vllm_policy import VLLMPolicy  # noqa: E402
from srd.eval.runner import shard  # noqa: E402
from srd.prompts import Prompts  # noqa: E402
from srd.protocol import MASKED_TAGS, ProtoConfig  # noqa: E402
from srd.reward import RewardConfig, r_ans  # noqa: E402
from srd.teacher import collect, merge, quotas, shard_quotas, split_total, teacher_ok  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def _flag(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def trace_name(mode: str, val: bool) -> str:
    if mode == "cot":
        return "traces_cot"
    if val:
        return "traces_val" if mode == "protocol" else f"traces_{mode}_val"
    return f"traces_{mode}"


def merge_shards(out: Path, name: str, total: int, mix: dict | None) -> int:
    files = sorted(out.glob(f"{name}.shard*of*.jsonl"))
    if not files:
        raise SystemExit(f"no shard files {name}.shard*of*.jsonl under {out}")
    recs, stats = merge([list(read_jsonl(f)) for f in files], quotas(total, mix) if mix else None, total)
    write_jsonl(out / f"{name}.jsonl", recs)
    stats["files"] = [f.name for f in files]
    write_json(out / f"{name}.stats.json", stats)
    log(f"merged {len(files)} files -> {name}.jsonl: {stats['got']} (want {stats['want']})")
    print("TEACHER_MERGE_DONE")
    return 0


def main() -> int:
    cfg, flags = config_from_argv()
    t = to_plain(cfg.teacher)
    out = Path(t["out"])
    mode = _flag(flags, "--mode", "protocol")
    val_n = _flag(flags, "--val")
    name = trace_name(mode, bool(val_n))
    total = int(val_n) if val_n else t["total"]
    if "--merge" in flags:
        return merge_shards(out, name, total, None if mode == "cot" else t["mix"])
    i, n = shard_args()
    if "--pilot" in flags and i > 0:
        print("TEACHER_PILOT_SKIPPED (shard > 0)")
        return 0
    ecfg = to_plain(cfg.engine)
    proto = ProtoConfig.from_cfg(to_plain(cfg.protocol))
    rcfg = RewardConfig.from_cfg(to_plain(cfg.reward))
    prompts = Prompts.from_cfg(to_plain(cfg.get("prompts")))
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    stok = load_tok(cfg.student_tokenizer)
    count = (lambda x: len(stok.encode(x)))
    engine = make_vllm_engine(ecfg, False)
    temp, top_p = ecfg.get("temperature", 0.7), ecfg.get("top_p", 1.0)
    samples = load_samples(t["samples"] if not val_n else t["samples"].replace("train.jsonl", "val.jsonl"))
    if t.get("reserve") and not val_n:
        samples += load_samples(t["reserve"])
    if "--pilot" not in flags:
        samples = shard(samples, i, n)
    if mode == "cot":
        recs = []
        my_total = split_total(total, i, n)
        outs = run_plain(engine, tok, prompts, samples[: 2 * my_total], "plain", temperature=temp, n=t["attempts"])
        by = {s.qid: s for s in samples}
        seen = set()
        for o in outs:
            if o["qid"] in seen or o["truncated"] or r_ans(o["answer"], by[o["qid"]]) < 1:
                continue
            seen.add(o["qid"])
            recs.append({"qid": o["qid"], "text": o["text"], "source": "teacher_cot"})
            if len(recs) >= my_total:
                break
        write_jsonl(shard_suffix(out / f"{name}.jsonl"), recs)
        log(f"{len(recs)} CoT traces")
        print("TEACHER_DONE")
        return 0
    pol = VLLMPolicy(engine, tok, prompts, "masked" if mode == "rag" else "two_stream", proto, temp, top_p)
    drv = DriverConfig(proto=proto, locator="retriever" if mode == "rag" else "model")
    retr = make_retriever(to_plain(cfg.get("retriever"))) if mode == "rag" else None
    if "--pilot" in flags:
        import time

        ss = samples[: t["pilot"]]
        t0 = time.time()
        eps = run_episodes(pol, ss, 1, drv, retriever=retr)
        secs = time.time() - t0
        lens = defaultdict(list)
        why = defaultdict(int)
        for ep in eps:
            ok, w = teacher_ok(ep, proto, rcfg, None)
            why[w] += 1
            for st in ep.steps:
                if st.tag in MASKED_TAGS:
                    lens[st.tag].append(count(st.text))
        caps = {k: int(math.ceil(np.percentile(v, 99))) for k, v in lens.items() if v}
        rate = why.get("ok", 0) / max(1, len(eps))
        # one episode per question here; the real run tries up to 4 per question until one passes, all batched
        per_ep = secs / max(1, len(eps))
        est = {f"hours_for_{k}_traces": (round(k / rate * per_ep / 3600, 1) if rate else None) for k in (2000, 3000)}
        write_json(out / "pilot.json", {"caps_p99": caps, "filter": dict(why), "n": len(eps), "pass_rate": rate,
                                        "seconds": round(secs, 1), "seconds_per_episode": round(per_ep, 2),
                                        "same_gpu_estimate": est})
        log(f"pilot: caps (p99, student tokens) {caps}; filter {dict(why)}; pass rate {rate:.2f}; "
            f"{per_ep:.1f} s per episode -> {est}")
        print("TEACHER_PILOT_DONE")
        return 0
    want = shard_quotas(total, t["mix"], i, n)
    recs, stats = collect(pol, samples, total, t["mix"], drv, rcfg, t["attempts"], t["batch"], count, retr,
                          log=log, seed=cfg.run.seed * 1000 + i, want=want)
    path = shard_suffix(out / f"{name}.jsonl")
    write_jsonl(path, recs)
    write_json(path.with_suffix(".stats.json"), stats)
    log(json.dumps(stats))
    print("TEACHER_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
