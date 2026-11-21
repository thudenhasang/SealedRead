"""One round of filtered self-training (ReST-EM, section 5.4) and the variants of RQ3.

For each of the 7,000 questions: N = 4 samples (T = 0.7) with the branch's own engine; hard filter (valid, right
answer, recall >= 0.5 when |G| >= 1); a question is "learnable" when 1-3 of its 4 samples pass: keep at most 2,
ranked by R_mu (ties: fewer IDs); a question where all 4 pass keeps 1. At most 4,000 samples, learnable questions
first. The M-step fine-tunes the S1 checkpoint on these samples only (no teacher traces mixed back).

RQ3 variants (re-filter the same round-1 samples of SealedRead seed 1):
    soft    recall >= 0.5, rank by R_mu with lambda = 0.2 (the main configuration)
    hard    recall >= 0.5, rank by R_mu with lambda = 0 (g = F2)
    nogate  no recall filter, rank by R_ans - mu * Acc (right-without-looking samples can enter)
docs/decisions.md D6: with the recall filter in place, soft and hard only differ in tie-breaking among <= 4 samples.
"""
from __future__ import annotations

import random
from collections import defaultdict

from srd.engine.driver import Episode
from srd.protocol import ProtoConfig, steps_to_dicts
from srd.reward import RewardConfig, score

VARIANTS = ("soft", "hard", "nogate")


def passes(ep: Episode, proto: ProtoConfig, rcfg: RewardConfig, variant: str = "soft", recall_min: float = 0.5,
           token_count=None) -> bool:
    if ep.violations(proto, token_count):
        return False
    sc = score(ep.steps, ep.sample, rcfg)
    if sc.r_ans < 1.0:
        return False
    if variant != "nogate" and ep.sample.gold and sc.recall < recall_min:
        return False
    return True


def rank_key(ep: Episode, rcfg: RewardConfig, variant: str) -> tuple[float, int]:
    if variant == "nogate":
        sc = score(ep.steps, ep.sample, rcfg, gate=False)
    else:
        sc = score(ep.steps, ep.sample, rcfg, lam=0.0 if variant == "hard" else rcfg.lam)
    return (-(sc.r_mu or 0.0), sc.n_id)


def select(episodes: list[Episode], proto: ProtoConfig, rcfg: RewardConfig, variant: str = "soft", n: int = 4,
           keep: int = 2, cap: int = 4000, recall_min: float = 0.5, seed: int = 0, token_count=None) -> tuple[list[dict], dict]:
    by_q: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        by_q[ep.sample.qid].append(ep)
    learnable, easy = [], []
    stats = defaultdict(int)
    for qid, eps in by_q.items():
        ok = [e for e in eps if passes(e, proto, rcfg, variant, recall_min, token_count)]
        stats[f"pass_{len(ok)}"] += 1
        if not ok:
            continue
        ok.sort(key=lambda e: rank_key(e, rcfg, variant))
        if len(ok) < len(eps):
            learnable.append(ok[:keep])
        else:
            easy.append(ok[:1])
    rng = random.Random(seed)
    rng.shuffle(learnable)
    rng.shuffle(easy)
    chosen: list[Episode] = []
    for group in learnable + easy:
        if len(chosen) + len(group) > cap:
            if len(chosen) >= cap:
                break
            group = group[: cap - len(chosen)]
        chosen.extend(group)
    recs = []
    for e in chosen:
        sc = score(e.steps, e.sample, rcfg)
        recs.append({"qid": e.sample.qid, "steps": steps_to_dicts(e.steps), "source": f"rest_em:{variant}",
                     "r_mu": sc.r_mu, "recall": sc.recall, "n_id": sc.n_id, "j": e.sample_idx})
    stats.update(questions=len(by_q), learnable=len(learnable), all_pass=len(easy), kept=len(recs))
    return recs, dict(stats)
