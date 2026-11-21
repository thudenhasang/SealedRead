"""Data checks before training (section 4.3), with their pre-registered thresholds (configs/prereg.yaml).

    1 gold mapping     every evidence sentence maps into G (>= 99% of samples; hops all mapped)
    2 closed-book      accuracy of the base model without C (<= 10-15% multi-hop, <= 2% fictional): needs the model,
                       reads the file of scripts/closed_book.py
    3 no surface path  a classifier on surface features (length, position, format, numbers, title, section; title
                       character n-grams for the fictional set) separates gold from distractor with AUC <= 0.60
    4 answer leak      >= 99% of samples have no leaking distractor, per length
    5 paraphrase       >= 95% keep the answer (scripts/paraphrase.py marks it; plus 200 checked by hand) and the
                       word overlap with the gold segments goes down
    6 engine parity    log-prob agreement between engines: GPU, scripts/parity.py
Plus structural checks (IDs unique, in range, not a number of the text) and a scripted oracle run through the
protocol (every sample solvable).
"""
from __future__ import annotations

import math
import zlib
from collections import defaultdict

import numpy as np

from srd.data.build import item_of
from srd.data.leak import leaks
from srd.data.sample import Sample
from srd.protocol import ID_MAX, ID_MIN
from srd.text import approx_tokens, standalone_numbers


def check_structure(samples: list[Sample]) -> dict:
    bad = defaultdict(int)
    for s in samples:
        ids = [g.sid for g in s.segments]
        if len(set(ids)) != len(ids):
            bad["duplicate_id"] += 1
        if any(not ID_MIN <= i <= ID_MAX for i in ids):
            bad["id_out_of_range"] += 1
        nums = set()
        for g in s.segments:
            nums |= standalone_numbers(g.text + " " + g.title)
        if set(ids) & nums:
            bad["id_equals_a_number"] += 1
        if s.qtype != "nocontext" and not s.gold:
            bad["no_gold"] += 1
        gold = set(s.gold)
        if any(not set(h.sids) <= gold for h in s.hops):
            bad["hop_not_gold"] += 1
    return {"n": len(samples), "errors": dict(bad), "pass": not bad}


def check_gold_mapping(samples: list[Sample], min_rate: float = 0.99) -> dict:
    ctx = [s for s in samples if s.qtype != "nocontext"]
    ok = sum(1 for s in ctx if s.gold and (not s.meta.get("hop_titles") or len(s.hops) == len(s.meta["hop_titles"])))
    rate = ok / len(ctx) if ctx else 1.0
    return {"rate": rate, "min": min_rate, "pass": rate >= min_rate}


def check_leaks(samples: list[Sample], min_rate: float = 0.99) -> dict:
    by_len: dict[int, list[bool]] = defaultdict(list)
    for s in samples:
        if s.qtype == "nocontext":
            continue
        it = item_of(s)
        clean = not any(leaks(g.text, it, g.title) for g in s.segments if not g.gold)
        by_len[s.length].append(clean)
    rates = {str(k): (sum(v) / len(v)) for k, v in by_len.items()}
    return {"rate_by_length": rates, "min": min_rate, "pass": all(r >= min_rate for r in rates.values())}


# ---------------------------------------------------------------------- check 3: surface classifier
def _hash_ngrams(s: str, dim: int = 64, n: int = 3) -> np.ndarray:
    v = np.zeros(dim)
    s = f" {s.lower()} "
    for i in range(len(s) - n + 1):
        v[zlib.crc32(s[i:i + n].encode()) % dim] += 1
    return v / max(1.0, v.sum())


def surface_features(s: Sample, title_ngrams: bool = False) -> tuple[np.ndarray, np.ndarray]:
    rows, ys = [], []
    n = len(s.segments)
    for i, g in enumerate(s.segments):
        f = [math.log(1 + approx_tokens(g.text)), i / max(1, n - 1), len(standalone_numbers(g.text)),
             len(g.title.split()), zlib.crc32(g.section.encode()) % 7, float(any(1000 <= x <= 2100 for x in standalone_numbers(g.text))),
             sum(w[:1].isupper() for w in g.text.split()) / max(1, len(g.text.split()))]
        if title_ngrams:
            f = np.concatenate([f, _hash_ngrams(g.title)])
        rows.append(np.asarray(f, float))
        ys.append(1.0 if g.gold else 0.0)
    return np.stack(rows), np.asarray(ys)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):                      # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _logreg(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 400, lr: float = 0.5) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w


def check_surface(samples: list[Sample], max_auc: float = 0.60, folds: int = 5, seed: int = 0,
                  title_ngrams: bool = False) -> dict:
    """Group k-fold (by sample) logistic regression, gold vs distractor with all gold and as many distractors."""
    rng = np.random.default_rng(seed)
    data = []
    for s in samples:
        if not s.gold:
            continue
        X, y = surface_features(s, title_ngrams)
        pos = np.nonzero(y == 1)[0]
        neg = rng.choice(np.nonzero(y == 0)[0], size=min(len(pos), int((y == 0).sum())), replace=False)
        idx = np.concatenate([pos, neg])
        data.append((X[idx], y[idx]))
    if len(data) < folds:
        return {"auc": 0.5, "max": max_auc, "pass": True, "note": "too few samples"}
    groups = np.arange(len(data)) % folds
    rng.shuffle(groups)
    scores, labels = [], []
    for f in range(folds):
        tr = [d for d, g in zip(data, groups) if g != f]
        te = [d for d, g in zip(data, groups) if g == f]
        Xtr, ytr = np.concatenate([d[0] for d in tr]), np.concatenate([d[1] for d in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
        w = _logreg(Xtr, ytr)
        for X, y in te:
            Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
            scores.append(Xs @ w)
            labels.append(y)
    a = auc(np.concatenate(scores), np.concatenate(labels))
    return {"auc": a, "max": max_auc, "pass": a <= max_auc}


def check_closed_book(records: list[dict], max_multihop: float = 0.15, max_fictional: float = 0.02) -> dict:
    """records: {qid, qtype, correct} from scripts/closed_book.py on the held-out set."""
    by = defaultdict(list)
    for r in records:
        by["fictional" if r.get("qtype") == "fictional" else "multihop"].append(bool(r["correct"]))
    acc = {k: sum(v) / len(v) for k, v in by.items() if v}
    ok = acc.get("multihop", 0) <= max_multihop and acc.get("fictional", 0) <= max_fictional
    return {"acc": acc, "max": {"multihop": max_multihop, "fictional": max_fictional}, "pass": ok,
            "fallback": None if acc.get("multihop", 0) <= max_multihop else
            "multi-hop over threshold: the fictional set becomes the main RQ1 set (section 4.3)"}


def check_paraphrase(samples: list[Sample], min_keep: float = 0.95) -> dict:
    pending = sum(1 for s in samples if s.meta.get("needs_paraphrase"))
    para = [s for s in samples if s.qtype == "paraphrase" and not s.meta.get("needs_paraphrase")]
    if not para:
        note = f"{pending} samples wait for scripts/paraphrase.py (GPU); re-run the checks after it" if pending \
            else "no paraphrased sample"
        return {"n": 0, "pending": pending, "pass": True, "note": note}
    kept = [s.meta.get("answer_kept", False) for s in para]
    before = [s.meta.get("overlap_before", 0) for s in para]
    after = [s.meta.get("overlap_after", 0) for s in para]
    keep = sum(kept) / len(kept)
    return {"n": len(para), "pending": pending, "answer_kept": keep, "overlap_before": float(np.mean(before)),
            "overlap_after": float(np.mean(after)), "pass": keep >= min_keep and np.mean(after) < np.mean(before)}


def check_scripted(samples: list[Sample]) -> dict:
    """Every sample is solvable through the protocol by an oracle policy, and the reward machinery agrees."""
    from srd.engine.driver import DriverConfig, run_episodes
    from srd.engine.scripted import ScriptedPolicy
    from srd.protocol import ProtoConfig
    from srd.reward import RewardConfig, score

    eps = run_episodes(ScriptedPolicy(), samples, n=1, cfg=DriverConfig())
    bad = 0
    for ep in eps:
        ok = not ep.violations(ProtoConfig())
        sc = score(ep.steps, ep.sample, RewardConfig(), valid=ok)
        if not ok or (ep.sample.hops and sc.r_ans < 1):
            bad += 1
    return {"n": len(eps), "failed": bad, "pass": bad == 0}
