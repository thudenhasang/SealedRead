"""On the real model: the single chain (C present, masked) and the masked view in the full frame (C absent, positions
jumping over |C|) must give the same masked-phase log-probs. This is what lets S0 config (b) run without C. The CPU
tests check it on a tiny random model; this checks it with the real weights and kernels (bf16, FlexAttention).

    python scripts/equivalence_check.py configs/eval.yaml --samples <train.jsonl> --traces <traces.jsonl> [--n 10]
Pass: the worst p99 |Delta log p| over the traces <= max(2 x the worst numerical-noise p99, 0.02).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import config_from_argv, log, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples, read_jsonl  # noqa: E402
from srd.engine.factory import load_ref_model  # noqa: E402
from srd.eval.probes import chain_vs_masked  # noqa: E402
from srd.prompts import Prompts  # noqa: E402
from srd.protocol import steps_from_dicts  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def _flag(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def main() -> int:
    cfg, flags = config_from_argv()
    ecfg = to_plain(cfg.engine)
    prompts = Prompts.from_cfg(to_plain(cfg.get("prompts")))
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    model = load_ref_model(ecfg)
    mode = "flex" if model.device.type == "cuda" else "dense"
    pad = 1024 if mode == "flex" else 1
    by = {s.qid: s for s in load_samples(_flag(flags, "--samples"))}
    traces = [t for t in read_jsonl(_flag(flags, "--traces")) if t["qid"] in by][: int(_flag(flags, "--n", 10))]
    rows = []
    for k, t in enumerate(traces):
        rows.append(chain_vs_masked(model, by[t["qid"]], steps_from_dicts(t["steps"]), tok, prompts, mode, pad))
        log(f"{k + 1}/{len(traces)}: p99 {rows[-1]['p99']:.4f} (noise {rows[-1]['noise_p99']:.4f})")
    worst, noise = max(r["p99"] for r in rows), max(r["noise_p99"] for r in rows)
    ok = worst <= max(2 * noise, 0.02)
    res = {"n": len(rows), "worst_p99": worst, "worst_noise_p99": noise, "pass": ok, "rows": rows}
    write_json(Path(cfg.run.out_dir) / "equivalence.json", res)
    print("EQUIV_PASS" if ok else "EQUIV_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
