"""Process reward R_mu (section 5.1, equations 2, 4, 5) and the answer score R_ans.

    R_mu = R_ans * g(F2) - mu * Acc          when the sample needs context (|G| >= 1)
    R_mu = R_ans - mu0 * min(n_loc, 3)       when it does not (|G| = 0)
    g(r) = lambda + (1 - lambda) * r,  F2 = 5PR / (4P + R),  Acc = n_ID / (K * R_max)

In the single ReST-EM round the reward only ranks the (at most 4) samples of a question that passed the hard
filter; see docs/decisions.md D6 for what this implies for RQ3.
"""
from __future__ import annotations

from dataclasses import dataclass

from srd.data.sample import Sample
from srd.protocol import Step, answer_of, located, rounds
from srd.text import best_f1, exact_match, set_f1


@dataclass
class RewardConfig:
    lam: float = 0.2
    mu: float = 0.25
    mu0: float = 0.25
    k: int = 3
    r_max: int = 8

    @staticmethod
    def from_cfg(d: dict | None) -> "RewardConfig":
        d = dict(d or {})
        return RewardConfig(**{k: v for k, v in d.items() if k in ("lam", "mu", "mu0", "k", "r_max")})


def r_ans(pred: str, s: Sample) -> float:
    """Exact match after normalization; set F1 for list answers."""
    if s.answer_kind == "list":
        return set_f1(pred, s.answers)
    return float(exact_match(pred, s.answers))


def answer_f1(pred: str, s: Sample) -> float:
    if s.answer_kind == "list":
        return set_f1(pred, s.answers)
    return best_f1(pred, s.answers)


def precision_recall(gold: list[int], got: list[int]) -> tuple[float, float]:
    g, s = set(gold), set(got)
    if not s or not g:
        return 0.0, 0.0
    tp = len(g & s)
    return tp / len(s), tp / len(g)


def f2(gold: list[int], got: list[int]) -> float:
    p, r = precision_recall(gold, got)
    if p == 0 and r == 0:
        return 0.0
    return 5 * p * r / (4 * p + r)


def acc_cost(n_ids: int, cfg: RewardConfig) -> float:
    return n_ids / (cfg.k * cfg.r_max)


@dataclass
class Score:
    valid: bool
    r_ans: float
    f2: float
    precision: float
    recall: float
    n_id: int
    n_loc: int
    acc: float
    r_mu: float | None      # None for invalid solutions (not scored)


def score(steps: list[Step], s: Sample, cfg: RewardConfig, valid: bool = True, lam: float | None = None,
          gate: bool = True) -> Score:
    """All reward terms of one solution. `gate=False`: the no-gate variant of RQ3 (R_ans - mu * Acc)."""
    lam = cfg.lam if lam is None else lam
    ids = located(steps)
    gold = s.gold
    n_loc = rounds(steps)
    ra = r_ans(answer_of(steps), s) if valid else 0.0
    p, r = precision_recall(gold, ids)
    f = f2(gold, ids)
    acc = acc_cost(len(ids), cfg)
    if not valid:
        rm = None
    elif not gold:
        rm = ra - cfg.mu0 * min(n_loc, 3)
    elif not gate:
        rm = ra - cfg.mu * acc
    else:
        rm = ra * (lam + (1 - lam) * f) - cfg.mu * acc
    return Score(valid, ra, f, p, r, len(ids), n_loc, acc, rm)
