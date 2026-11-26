"""RQ4 (a)/(b): invariance and dependence probes on trajectories from evaluation records (GPU, our Qwen3 +
FlexAttention; dense on CPU for tests).

    (--records: evaluation records with steps, or a trace file such as the output of make_scripted_traces.py)
    python scripts/probes.py configs/eval.yaml --records /content/runs/eval/sealedread__test_64k.jsonl \
        --samples /content/data/srd/test_64k.jsonl --branch sealedread [--n 200] engine.adapter=...
Branches: sealedread (expected invariant: an implementation test), nomask and daft (the leak they measure).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from srd.cli import config_from_argv, log, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples  # noqa: E402
from srd.engine.factory import load_ref_model  # noqa: E402
from srd.eval.probes import dependence, invariance  # noqa: E402
from srd.eval.runner import load_records  # noqa: E402
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
    branch = _flag(flags, "--branch", "sealedread")
    by = {s.qid: s for s in load_samples(_flag(flags, "--samples"))}
    # evaluation records, or plain trace files (make_scripted_traces.py) that carry no "valid" field
    recs = [r for r in load_records(_flag(flags, "--records")) if r.get("valid", True) and r.get("steps")]
    recs = recs[: int(_flag(flags, "--n", 200))]
    out = Path(_flag(flags, "--out", str(Path(cfg.eval.out) / f"probes_{branch}.jsonl")))
    rows = []
    for k, r in enumerate(recs):
        s, steps = by[r["qid"]], steps_from_dicts(r["steps"])
        row = invariance(model, s, steps, tok, prompts, branch, mode, pad_to=1024 if mode == "flex" else 1, seed=k)
        dep = dependence(model, s, steps, tok, prompts, "Zorvath", branch, mode, 1024 if mode == "flex" else 1)
        row["dependence"] = dep
        rows.append(row)
        if (k + 1) % 20 == 0:
            log(f"{k + 1}/{len(recs)} probed")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    kl = [x["kl_max"] for x in rows]
    noise = [x["noise_kl_max"] for x in rows]
    drops = [x["dependence"]["drop"] for x in rows if x["dependence"]]
    summ = {"branch": branch, "n": len(rows), "kl_max_median": float(np.median(kl)) if kl else None,
            "kl_max_p99": float(np.percentile(kl, 99)) if kl else None,
            "noise_kl_max_p99": float(np.percentile(noise, 99)) if noise else None,
            "invariant_rate": float(np.mean([a <= max(b, 1e-6) * 1.5 for a, b in zip(kl, noise)])) if kl else None,
            "dependence_drop_median": float(np.median(drops)) if drops else None}
    write_json(out.with_suffix(".summary.json"), summ)
    log(json.dumps(summ))
    if branch in ("sealedread", "sealedread_chain"):        # an implementation test: must be invariant
        print("INVARIANCE_PASS" if summ["invariant_rate"] == 1.0 else "INVARIANCE_FAILED (a mask bug)")
    else:
        print(f"LEAK_MEASURED {branch}: invariant on {summ['invariant_rate']:.0%} of the traces (expected: low)")
    print("PROBES_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
