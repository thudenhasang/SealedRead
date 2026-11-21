"""Per-episode metrics and their aggregates (section 6.2).

answer        EM, token F1 (set F1 for lists)
locate        precision / recall of all located IDs against G; "right without looking" (recall 0, answer right)
tightness     done by probes (invariance KL, dependence) — srd.eval.probes
ID channel    hop-skip rate, UCR, accuracy under ID permutation (driver option)
behaviour     rounds, R_max hit rate, tokens per tag, Acc, rate of opening the context on |G| = 0 samples
Open vs closed answers are reported apart for "right without looking" and UCR: guessing a closed answer is easy.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from srd.data.sample import Sample
from srd.engine.driver import Episode
from srd.prompts import Prompts
from srd.protocol import ProtoConfig, Step, answer_of, rounds
from srd.reward import RewardConfig, answer_f1, score
from srd.text import contains


def visible_text(steps: list[Step], s: Sample, prompts: Prompts) -> str:
    """Everything the masked phase could read: its own text and the pasted segments."""
    parts = []
    for st in steps:
        if st.tag == "knowledge":
            parts.append(prompts.knowledge_body(s, st.ids))
        elif st.tag != "locate":
            parts.append(st.text)
    return "\n".join(parts)


def hop_skip(steps: list[Step], s: Sample, prompts: Prompts) -> bool | None:
    """A trajectory skips a hop if it locates a segment of step j >= 2 before the bridge of step j-1 appeared in
    the masked-phase text (its own writing + pasted text). None when the sample has no ordered hops."""
    if len(s.hops) < 2:
        return None
    step_of: dict[int, int] = {}
    for j, h in enumerate(s.hops):
        for sid in h.sids:
            step_of.setdefault(sid, j)
    for i, st in enumerate(steps):
        if st.tag != "locate":
            continue
        seen = visible_text(steps[:i], s, prompts)
        for sid in st.ids:
            j = step_of.get(sid)
            if j is not None and j >= 1 and not contains(seen, s.hops[j - 1].bridge):
                return True
    return False


def unsupported(steps: list[Step], s: Sample, prompts: Prompts) -> bool:
    """UCR numerator: right answer whose (normalized) string is absent from every pasted text."""
    pasted = "\n".join(prompts.knowledge_body(s, st.ids) for st in steps if st.tag == "knowledge")
    return not any(contains(pasted, a) for a in s.answers)


def episode_record(ep: Episode, proto: ProtoConfig, rcfg: RewardConfig, prompts: Prompts | None = None,
                   token_count=None, extra: dict | None = None) -> dict:
    prompts = prompts or Prompts()
    s = ep.sample
    viol = ep.violations(proto, token_count)
    valid = not viol
    sc = score(ep.steps, s, rcfg, valid=True)
    ans = answer_of(ep.steps)
    correct = sc.r_ans >= 1.0
    rec = {
        "qid": s.qid, "cluster": s.cluster, "length": s.length, "qtype": s.qtype, "answer_kind": s.answer_kind,
        "dataset": s.dataset, "j": ep.sample_idx, "valid": valid, "violations": viol[:3], "answer": ans,
        "em": float(correct and valid), "f1": answer_f1(ans, s) if valid else 0.0, "r_ans": sc.r_ans if valid else 0.0,
        "precision": sc.precision, "recall": sc.recall, "f2": sc.f2, "n_id": sc.n_id, "n_loc": sc.n_loc,
        "acc": sc.acc, "r_mu": score(ep.steps, s, rcfg, valid=valid).r_mu, "rounds": rounds(ep.steps),
        "hit_rmax": rounds(ep.steps) >= proto.r_max, "n_gold": len(s.gold),
        "right_without_looking": bool(s.gold) and correct and sc.recall == 0.0,
        "opened_nocontext": (not s.gold) and sc.n_loc > 0,
        "hop_skip": hop_skip(ep.steps, s, prompts),
        "unsupported": (correct and s.answer_kind == "open" and bool(s.gold) and unsupported(ep.steps, s, prompts)),
        "stats": dict(ep.stats),
    }
    if token_count is not None:
        per = defaultdict(int)
        for st in ep.steps:
            if st.tag in ("reason", "query", "recheck", "answer"):
                per[st.tag] += token_count(st.text)
        rec["tokens"] = dict(per)
    if extra:
        rec.update(extra)
    return rec


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(records: list[dict]) -> dict:
    """Means over records (several samples per question count once each: average by question first)."""
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_q[f"{r['qid']}|{r['length']}"].append(r)
    keys = ("valid", "em", "f1", "precision", "recall", "n_id", "n_loc", "acc", "rounds", "hit_rmax")
    out = {"n_questions": len(by_q), "n_records": len(records)}
    for k in keys:
        out[k] = _mean(_mean(r[k] for r in rs) for rs in by_q.values())
    ctx = [r for r in records if r["n_gold"]]
    for kind in ("open", "closed"):
        sub = [r for r in ctx if r["answer_kind"] == kind]
        right = [r for r in sub if r["em"] >= 1]
        out[f"right_without_looking_{kind}"] = _mean(r["right_without_looking"] for r in right) if right else None
        out[f"ucr_{kind}"] = _mean(r["unsupported"] for r in right) if right else None
    out["hop_skip"] = _mean(r["hop_skip"] for r in records if r["hop_skip"] is not None)
    noc = [r for r in records if not r["n_gold"]]
    out["opened_nocontext"] = _mean(r["opened_nocontext"] for r in noc) if noc else None
    return out


def by_question(records: list[dict], key: str = "em") -> dict[str, tuple[str, float]]:
    """{qid|length: (cluster, mean of key over the question's records)} — the unit of the paired bootstrap."""
    acc: dict[str, list] = defaultdict(list)
    cl: dict[str, str] = {}
    for r in records:
        k = f"{r['qid']}|{r['length']}"
        acc[k].append(float(r[key]))
        cl[k] = r["cluster"]
    return {k: (cl[k], float(np.mean(v))) for k, v in acc.items()}
