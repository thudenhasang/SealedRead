"""Teacher traces for S1 (section 5.3): the teacher sees exactly what the student will see.

The protocol runs in two kinds of calls (the two-stream policy): the masked steps are generated in calls that hold
no C (instruction, question, clean history); <locate> is generated in calls that hold C and the history, with the ID
grammar. The reasoning steps therefore depend on text only, which reproduces the student's view without any mask
inside the teacher (and its Gated DeltaNet layers cannot peek at C).

Filter (all three): right answer; recall >= 0.5, or no context opened when |G| = 0; 100% valid protocol, with the
per-tag caps and the <locate> safety cap re-checked with the *student's* tokenizer. Up to 4 samples per question
(T = 0.7) until one passes; quotas per data type (table 4), shortfalls topped up from a reserve.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from srd.data.sample import Sample
from srd.engine.driver import DriverConfig, Episode, run_episodes
from srd.protocol import ProtoConfig, steps_to_dicts
from srd.reward import RewardConfig, score


def teacher_ok(ep: Episode, proto: ProtoConfig, rcfg: RewardConfig, student_count=None) -> tuple[bool, str]:
    viol = ep.violations(proto, student_count)
    if viol:
        return False, "invalid"
    sc = score(ep.steps, ep.sample, rcfg)
    if sc.r_ans < 1:
        return False, "wrong"
    if ep.sample.gold and sc.recall < 0.5:
        return False, "low_recall"
    if not ep.sample.gold and sc.n_loc > 0:
        return False, "opened_without_need"
    return True, "ok"


def quotas(total: int, mix: dict[str, float]) -> dict[str, int]:
    q = {k: int(round(total * v)) for k, v in mix.items()}
    diff = total - sum(q.values())
    if diff:
        q[max(mix, key=mix.get)] += diff
    return q


def split_total(total: int, i: int, n: int) -> int:
    """Share i of n of an integer (the shares sum to total)."""
    return total // n + (1 if i < total % n else 0)


def shard_quotas(total: int, mix: dict[str, float], i: int, n: int) -> dict[str, int]:
    """The global quota of every type, split across n teacher processes (shares sum to the global quotas)."""
    return {t: split_total(q, i, n) for t, q in quotas(total, mix).items()}


def merge(parts: list[list[dict]], want: dict[str, int] | None, total: int) -> tuple[list[dict], dict]:
    """Merge the trace files of several teacher processes; keep each type within its global quota."""
    got: Counter = Counter()
    seen: set[str] = set()
    out: list[dict] = []
    for recs in parts:
        for r in recs:
            t = r.get("qtype", "all")
            cap = want.get(t, 0) if want else total
            if r["qid"] in seen or got[t] >= cap or len(out) >= total:
                continue
            seen.add(r["qid"])
            got[t] += 1
            out.append(r)
    return out, {"want": want or {"all": total}, "got": dict(got), "total": len(out)}


def collect(policy, samples: list[Sample], total: int, mix: dict[str, float], drv: DriverConfig, rcfg: RewardConfig,
            attempts: int = 4, batch: int = 64, student_count=None, retriever=None, seed: int = 0,
            log=print, want: dict[str, int] | None = None) -> tuple[list[dict], dict]:
    """Run the teacher over `samples` (shuffled) until every type's quota is filled or samples run out.
    `want`: explicit quotas (one share of the global quotas when several teacher processes run in parallel)."""
    want = dict(want) if want is not None else quotas(total, mix)
    total = sum(want.values())
    got: Counter = Counter()
    reasons: Counter = Counter()
    per_type_tries: dict[str, int] = defaultdict(int)
    rng = random.Random(seed)
    pending = list(samples)
    rng.shuffle(pending)
    out: list[dict] = []
    while pending and any(got[t] < want[t] for t in want):
        chunk, rest = [], []
        for s in pending:
            t = _type(s)
            if t in want and got[t] + sum(1 for c in chunk if _type(c) == t) < want[t] and len(chunk) < batch:
                chunk.append(s)
            else:
                rest.append(s)
        pending = [s for s in rest if got[_type(s)] < want.get(_type(s), 0)]
        if not chunk:
            break
        eps = run_episodes(policy, chunk, attempts, drv, retriever=retriever, seed=seed + len(out))
        for i, s in enumerate(chunk):
            t = _type(s)
            per_type_tries[t] += 1
            for ep in eps[i * attempts:(i + 1) * attempts]:
                ok, why = teacher_ok(ep, drv.effective_proto(), rcfg, student_count)
                reasons[why] += 1
                if ok and got[t] < want[t]:
                    sc = score(ep.steps, ep.sample, rcfg)
                    out.append({"qid": s.qid, "qtype": t, "steps": steps_to_dicts(ep.steps), "source": "teacher",
                                "recall": sc.recall, "n_id": sc.n_id, "rounds": sc.n_loc})
                    got[t] += 1
                    break
        log(f"teacher: {len(out)}/{total} traces " + " ".join(f"{t}={got[t]}/{want[t]}" for t in want))
    stats = {"want": want, "got": dict(got), "tries": dict(per_type_tries), "filter": dict(reasons),
             "pass_rate": {t: got[t] / max(1, per_type_tries[t]) for t in want}}
    return out, stats


def _type(s: Sample) -> str:
    return s.qtype if s.qtype in ("multihop", "paraphrase", "fictional", "multiseg", "nocontext") else "multihop"
