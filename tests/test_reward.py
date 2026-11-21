"""Table 5 of the proposal, row by row (lambda = 0.2, mu = mu0 = 0.25, K * R_max = 24)."""
from __future__ import annotations

import pytest

from conftest import make_sample
from srd.protocol import Step
from srd.reward import RewardConfig, score

CFG = RewardConfig()


def _steps(rounds: list[list[int]], answer: str) -> list[Step]:
    out = [Step("reason", "alpha=query")] if rounds else []
    for i, ids in enumerate(rounds):
        out += [Step("query", "q"), Step("locate", ids=ids), Step("knowledge", ids=ids, by="engine"),
                Step("recheck", "beta=valid_to_think"),
                Step("reason", "alpha=answer" if i == len(rounds) - 1 else "alpha=query")]
    if not rounds:
        out.append(Step("reason", "alpha=answer"))
    return out + [Step("answer", answer)]


@pytest.fixture
def ids():
    s = make_sample(n=40)
    noise = [x.sid for x in s.segments if not x.gold]
    return s, s.gold, noise


@pytest.mark.parametrize("name,rounds_fn,answer,f2,acc,r_mu", [
    ("both gold, 2 IDs", lambda g, n: [[g[0], g[1]]], "Porto", 1.00, 0.083, 0.98),
    ("both gold + 1 noise", lambda g, n: [[g[0], g[1], n[0]]], "Porto", 0.91, 0.125, 0.90),
    ("half the gold, 2 IDs", lambda g, n: [[g[0], n[0]]], "Porto", 0.50, 0.083, 0.58),
    ("right without looking", lambda g, n: [[n[0], n[1]]], "Porto", 0.0, 0.083, 0.18),
    ("all evidence, wrong answer", lambda g, n: [[g[0], g[1], n[0]]], "Lisbon", 0.91, 0.125, -0.03),
    ("right but 6 rounds, 15 IDs", lambda g, n: [[g[0], g[1], n[0]], n[1:4], n[4:7], n[7:10], n[10:12], [n[12]]],
     "Porto", 0.43, 0.625, 0.39),
])
def test_table5_context_rows(ids, name, rounds_fn, answer, f2, acc, r_mu):
    s, g, n = ids
    sc = score(_steps(rounds_fn(g, n), answer), s, CFG)
    assert sc.f2 == pytest.approx(f2, abs=0.005), name
    assert sc.acc == pytest.approx(acc, abs=0.0005), name
    assert sc.r_mu == pytest.approx(r_mu, abs=0.005), name


def test_table5_nocontext_rows():
    s = make_sample(n=10, nocontext=True)
    assert score(_steps([], "42"), s, CFG).r_mu == pytest.approx(1.0)
    noise = [x.sid for x in s.segments]
    sc = score(_steps([[noise[0], noise[1]], [noise[2]]], "42"), s, CFG)
    assert sc.n_id == 3 and sc.r_mu == pytest.approx(0.5)


def test_invalid_is_not_scored(ids):
    s, g, n = ids
    assert score(_steps([[g[0]]], "Porto"), s, CFG, valid=False).r_mu is None


def test_break_even_probabilities():
    """Section 5.1: with recall only, an extra ID pays off at ~2.6% hit probability; with F2 at ~14%."""
    cost = CFG.mu / 24
    gain_recall = (1 - CFG.lam) * 0.5
    assert cost / gain_recall == pytest.approx(0.026, abs=0.001)
    # F2 from one gold out of 2 (P=1, R=.5): add a gold -> F2 = 1; add a noise -> P = R = .5
    f_now, f_hit, f_miss = 5 * .5 / 4.5, 1.0, 0.5
    up, down = (1 - CFG.lam) * (f_hit - f_now), (1 - CFG.lam) * (f_now - f_miss)
    p = (down + cost) / (up + down)
    assert p == pytest.approx(0.14, abs=0.005)
