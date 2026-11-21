"""Statistics of section 6.3: paired cluster bootstrap, sample size, standard errors.

Paired comparisons are on the same questions; a cluster is a root question with all its paraphrases and lengths
(multi-hop) or a document (real-document benchmarks). Seeds are averaged per question before bootstrapping, so the
CI covers question sampling only (seed variance of the M-step is reported apart; see docs/06-evaluation.md).
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import NormalDist

import numpy as np


def sample_size(d: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """n = (z_{1-alpha/2} + z_power)^2 d / delta^2 for a paired comparison with discordance rate d."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return math.ceil(z * z * d / (delta * delta))


def se_paired(d: float, n: int) -> float:
    return math.sqrt(d / n)


def noninferiority_n(d: float, margin: float, true_delta: float = 0.0, alpha: float = 0.05, power: float = 0.8) -> int:
    """Questions needed so that the CI lower bound clears -margin with the given power when the true effect is
    true_delta (one-sided at alpha/2, as with a two-sided 95% CI)."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    gap = true_delta + margin
    return math.ceil(z * z * d / (gap * gap))


def average_over(values: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) for k, v in values.items()}


def paired_bootstrap(a: dict[str, tuple[str, float]], b: dict[str, tuple[str, float]], reps: int = 10000,
                     seed: int = 0, ci: float = 0.95) -> dict:
    """Mean of (a - b) over common keys, with a cluster bootstrap CI. a, b: {key: (cluster, value)}."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("no common question")
    by_cl: dict[str, list[float]] = defaultdict(list)
    for k in keys:
        by_cl[a[k][0]].append(a[k][1] - b[k][1])
    clusters = list(by_cl)
    sums = np.array([sum(by_cl[c]) for c in clusters])
    counts = np.array([len(by_cl[c]) for c in clusters])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(clusters), size=(reps, len(clusters)))
    boots = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    diffs = [a[k][1] - b[k][1] for k in keys]
    return {"delta": float(np.mean(diffs)), "lo": float(lo), "hi": float(hi), "n": len(keys),
            "clusters": len(clusters), "discordance": float(np.mean([x != 0 for x in diffs]))}


def bootstrap_mean(a: dict[str, tuple[str, float]], reps: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    zero = {k: (c, 0.0) for k, (c, _) in a.items()}
    r = paired_bootstrap(a, zero, reps, seed, ci)
    return {"mean": r["delta"], "lo": r["lo"], "hi": r["hi"], "n": r["n"], "clusters": r["clusters"]}


def scale(r: dict, k: float = 100.0) -> dict:
    """Points (x100) for reporting."""
    return {key: (v * k if key in ("delta", "lo", "hi", "mean") else v) for key, v in r.items()}
