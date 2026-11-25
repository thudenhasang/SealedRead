"""The pre-registered decision rules (configs/prereg.yaml): P0 gate and table 6, S1 decision point, S2 gate,
RQ1 success / non-inferiority / fallback, RQ2, optional round 2. Inputs are per-question values
{qid|length: (cluster, value)} (srd.eval.metrics.by_question), values in [0, 1]; outputs are in points."""
from __future__ import annotations

from srd.eval.stats import paired_bootstrap, scale

Q = dict[str, tuple[str, float]]


def _merge(*dicts: Q) -> Q:
    out: Q = {}
    for d in dicts:
        out.update(d)
    return out


def _avg_seeds(seeds: list[Q]) -> Q:
    keys = set.intersection(*(set(s) for s in seeds))
    return {k: (seeds[0][k][0], sum(s[k][1] for s in seeds) / len(seeds)) for k in keys}


def p0_decision(f: Q, d: Q, b: Q | None, pre: dict) -> dict:
    """Main design = two streams, compact frame (docs/decisions.md D1). Configs with both lengths merged.

    Gate:        Delta_oracle = EM(f) - EM(d): oracle paste, C masked + compact frame (SealedRead as deployed)
                 against oracle paste without mask (the control); >= delta_oracle_min with CI lower bound > 0.
    Diagnostic:  Pi = EM(f) - EM(b): what the compact frame alone adds (b = same mask, full frame). Reported with
                 Delta_mask in the paper as the positional share of the joint effect; it does not change the plan.
    """
    p0 = pre["p0"]
    oracle = scale(paired_bootstrap(f, d))
    gate = oracle["delta"] >= p0["delta_oracle_min"] and oracle["lo"] > 0
    pi = scale(paired_bootstrap(f, b)) if b else None
    action = ("keep the design (two streams, compact frame); go to S1" if gate else
              "switch to the RQ1 fallback narrative now; provenance (RQ4) becomes the main axis")
    return {"delta_oracle": oracle, "gate_pass": gate, "pi": pi,
            "pi_large": bool(pi and pi["delta"] >= p0["pi_large"] and pi["lo"] > 0), "action": action}


def s1_decision(sr: Q, nm: Q, pre: dict) -> dict:
    r = scale(paired_bootstrap(sr, nm))
    skip = r["delta"] <= pre["s1"]["skip_nomask_rest_if_delta_le"] and r["hi"] < 0
    return {"delta_mask_sft": r, "skip_nomask_rest_em": skip,
            "action": "report RQ1 at SFT level (fallback); skip ReST-EM for the no-mask control" if skip
            else "continue as planned"}


def lora_vs_full(lora: Q, full: Q, pre: dict) -> dict:
    r = scale(paired_bootstrap(lora, full))
    switch = r["delta"] < -pre["s1"]["lora_vs_full_max_gap"] and r["hi"] < 0
    return {"lora_minus_full": r, "switch_to_full_ft": switch}


def s2_gate(valid_rate: float, sr: Q, config_a: Q, pre: dict) -> dict:
    r = scale(paired_bootstrap(sr, config_a))
    ok = valid_rate >= pre["s2_gate"]["valid_rate_min"] and r["delta"] >= 0
    return {"valid_rate": valid_rate, "sr_minus_a": r, "pass": ok}


def rq1(sr_seeds: list[Q], nm_seeds: list[Q], sr32_seeds: list[Q], nm32_seeds: list[Q], pre: dict) -> dict:
    p = pre["rq1"]
    main = scale(paired_bootstrap(_avg_seeds(sr_seeds), _avg_seeds(nm_seeds)))
    ni = scale(paired_bootstrap(_avg_seeds(sr32_seeds), _avg_seeds(nm32_seeds))) if sr32_seeds else None
    success = main["delta"] >= p["delta_mask_min"] and main["lo"] > 0 and \
        (ni is None or ni["lo"] > -p["noninferiority_margin"])
    lo, hi = p["fallback_band"]
    if success:
        claim = "masking helps the small model reason (RQ1 success)"
    elif lo < main["delta"] < hi:
        claim = ("fallback: masking does not lower accuracy, while giving structural provenance and the conditional "
                 "bottleneck, checked by RQ4")
    elif main["delta"] <= lo:
        claim = "negative result: report it, with Delta_ctx, Pi and the leak channels"
    else:
        claim = "Delta_mask >= +2 but the CI or the 32K non-inferiority check failed: report as not significant"
    return {"delta_mask": main, "noninferiority_32k": ni, "success": success, "claim": claim}


def rq2(sr: Q, rag: Q, pre: dict) -> dict:
    r = scale(paired_bootstrap(sr, rag))
    p = pre["rq2"]
    if r["delta"] >= p["delta_min"] and r["lo"] > 0:
        claim = "SealedRead beats the strongest RAG"
    elif abs(r["delta"]) < p["tie_band"]:
        claim = "tie: SealedRead matches a strong RAG, with structural provenance and the checked bottleneck"
    elif r["delta"] <= -p["tie_band"]:
        claim = "negative result against RAG"
    else:
        claim = "not significant"
    return {"sr_minus_rag": r, "claim": claim}


def round2(after: Q, before: Q, pre: dict) -> dict:
    r = scale(paired_bootstrap(after, before))
    return {"gain": r, "run_round2": r["delta"] >= pre["round2"]["em_gain_min"] and r["lo"] > 0}


def merge_lengths(*per_length: Q) -> Q:
    return _merge(*per_length)
