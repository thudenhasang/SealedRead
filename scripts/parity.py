"""Data check 6: log-prob agreement between engines (GPU), before most of the budget is spent.

Baseline: vLLM vs our Qwen3 on ordinary causal sequences (the locate view without hidden tokens). Then, on masked
sequences, FlexAttention (trainer) vs the dense reference (layout.visible); and vLLM vs our model on the masked
stream it serves. Pass: median |Delta log p| <= 1.5 x baseline and p99 <= 2 x baseline p99, with the p99 also
taken over boundary tokens (first token after a pasted block, copies, <locate>), where mask bugs show.

    python scripts/parity.py configs/eval.yaml --samples /content/data/srd/val.jsonl --n 50
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from srd.cli import config_from_argv, log, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples  # noqa: E402
from srd.engine.driver import DriverConfig, run_episodes  # noqa: E402
from srd.engine.factory import load_ref_model, make_vllm_engine  # noqa: E402
from srd.engine.scripted import ScriptedPolicy  # noqa: E402
from srd.layout import CPY, KNOW, LOC, Seq, build_seq  # noqa: E402
from srd.model.run import token_logprobs  # noqa: E402
from srd.prompts import Prompts  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def _flag(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def boundary(seq) -> np.ndarray:
    k = seq.kind
    first_after = np.nonzero((k[1:] != k[:-1]) & np.isin(k[:-1], [KNOW, CPY, LOC]))[0] + 1
    return np.union1d(first_after, np.nonzero(np.isin(k, [CPY, LOC]))[0])


def main() -> int:
    cfg, flags = config_from_argv()
    ecfg = to_plain(cfg.engine)
    prompts = Prompts.from_cfg(to_plain(cfg.get("prompts")))
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    ss = [s for s in load_samples(_flag(flags, "--samples"), int(_flag(flags, "--n", 50))) if s.hops]
    eps = run_episodes(ScriptedPolicy(noise=0.3), ss, 1, DriverConfig())
    traces = [(ep.sample, ep.steps) for ep in eps if not ep.error]
    plain = [build_seq(s, st, tok, prompts, view="locate") for s, st in traces]
    masked = [build_seq(s, st, tok, prompts, view="masked", frame="compact") for s, st in traces]
    chains = [build_seq(s, st, tok, prompts, view="chain", branch="sealedread") for s, st in traces]
    engine = make_vllm_engine({**ecfg, "gpu_memory_utilization": 0.45}, False)
    model = load_ref_model(ecfg)
    res = {}

    def vllm_vs_ours(seqs, name):
        diffs = []
        for sq in seqs:
            ids = sq.prompt_ids()
            vl = engine.prompt_logprobs([ids])[0]
            keep = sq.kind != LOC
            flat = Seq(**{**sq.__dict__, **{f: getattr(sq, f)[keep] for f in
                                            ("ids", "pos", "kind", "rnd", "seg", "open_at", "stp", "qref", "target")}})
            flat.pos = np.arange(len(flat.ids))
            idx = np.arange(1, len(flat.ids))
            ours = token_logprobs(model, flat, idx, "flex").cpu().numpy()
            v = np.array([x for x in vl[1:]], dtype=float)
            diffs.append(np.abs(ours - v))
        d = np.concatenate(diffs)
        res[name] = {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "n": int(d.size)}
        log(f"{name}: {res[name]}")

    vllm_vs_ours(plain, "vllm_vs_ours_plain")
    vllm_vs_ours(masked, "vllm_vs_ours_masked_stream")
    fl, bd = [], []
    for sq in chains:
        if len(sq) > 32768:
            continue
        idx = np.arange(1, len(sq))
        a = token_logprobs(model, sq, idx, "flex").cpu().numpy()
        b = token_logprobs(model, sq, idx, "dense").cpu().numpy()
        d = np.abs(a - b)
        fl.append(d)
        bd.append(d[boundary(sq)[boundary(sq) > 0] - 1])
    d, db = np.concatenate(fl), np.concatenate(bd)
    res["flex_vs_dense_chain"] = {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
                                  "p99_boundary": float(np.percentile(db, 99))}
    pre = yaml.safe_load(open("configs/prereg.yaml"))["data_checks"]["parity"]
    base = res["vllm_vs_ours_plain"]
    ok = all(res[k]["median"] <= pre["median_ratio_max"] * max(base["median"], 1e-6) and
             res[k]["p99"] <= pre["p99_ratio_max"] * max(base["p99"], 1e-6)
             for k in ("vllm_vs_ours_masked_stream", "flex_vs_dense_chain"))
    ok &= res["flex_vs_dense_chain"]["p99_boundary"] <= pre["p99_ratio_max"] * max(base["p99"], 1e-6)
    res["pass"] = bool(ok)
    write_json(Path(cfg.run.out_dir) / "parity.json", res)
    print("PARITY_PASS" if ok else "PARITY_FAILED")
    torch.cuda.empty_cache()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
