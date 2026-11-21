"""Driver + scripted policy; reference engine == trainer layout (forced and generated traces)."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import make_sample, trace_two_rounds
from srd.engine.driver import DriverConfig, Episode, run_episodes
from srd.engine.retriever import BM25Retriever
from srd.engine.scripted import ScriptedPolicy
from srd.layout import build_seq
from srd.protocol import ProtoConfig, answer_of
from srd.reward import RewardConfig, score


def _samples(k=4):
    return [make_sample(n=12, seed=i, qid=f"q{i}") for i in range(k)]


def test_scripted_solves_everything():
    eps = run_episodes(ScriptedPolicy(), _samples(), n=2, cfg=DriverConfig())
    assert len(eps) == 8
    for ep in eps:
        assert ep.violations(ProtoConfig()) == [], ep.steps
        assert answer_of(ep.steps) == "Porto"
        sc = score(ep.steps, ep.sample, RewardConfig())
        assert sc.recall == 1.0 and sc.r_ans == 1.0


def test_oracle_locator_pastes_gold_then_answers():
    eps = run_episodes(ScriptedPolicy(), _samples(2), n=1, cfg=DriverConfig(locator="oracle"))
    for ep in eps:
        kn = [s for s in ep.steps if s.tag == "knowledge"]
        assert len(kn) == 1 and set(kn[0].ids) == set(ep.sample.gold)
        assert ep.steps[-1].tag == "answer"


def test_retriever_locator():
    eps = run_episodes(ScriptedPolicy(), _samples(2), n=1, cfg=DriverConfig(locator="retriever"),
                       retriever=BM25Retriever())
    for ep in eps:
        assert all(s.by == "engine" for s in ep.steps if s.tag == "locate")
        assert ep.violations(ProtoConfig()) == []


def test_permuted_ids_are_fresh_and_same_length():
    eps = run_episodes(ScriptedPolicy(), _samples(2), n=1, cfg=DriverConfig(permute_ids=True))
    for ep in eps:
        for st in ep.steps:
            if st.tag in ("locate", "knowledge"):
                assert st.alias and set(st.alias) == set(st.ids)
                for a, b in st.alias.items():
                    assert len(str(a)) == len(str(b)) and b not in ep.sample.ids()


def test_nocontext_sample_answers_directly():
    s = make_sample(nocontext=True)
    eps = run_episodes(ScriptedPolicy(), [s], n=1, cfg=DriverConfig())
    assert [st.tag for st in eps[0].steps] == ["reason", "answer"]


def test_rmax_forces_an_answer():
    class Stubborn(ScriptedPolicy):
        def gen_masked(self, eps, tag, force):
            return [("more alpha=query", False) if tag == "reason" else (t, h)
                    for (t, h) in super().gen_masked(eps, tag, force)]

    eps = run_episodes(Stubborn(), _samples(1), n=1, cfg=DriverConfig(proto=ProtoConfig(r_max=3)))
    ep = eps[0]
    assert sum(1 for s in ep.steps if s.tag == "locate") <= 3 and ep.steps[-1].tag == "answer"
    assert ep.violations(ProtoConfig(r_max=3)) == []


# ---------------------------------------------------------------------- reference engine
torch = pytest.importorskip("torch")


def _tiny():
    from srd.model.qwen3 import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    m = Qwen3ForCausalLM(Qwen3Config.tiny()).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return m


def _same(a, b):
    for f in ("ids", "pos", "kind", "rnd", "open_at", "target"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f
    assert np.array_equal(a.mask_dense(), b.mask_dense())


@pytest.mark.parametrize("branch", ["sealedread", "nomask", "daft"])
def test_forced_trace_layout_equals_trainer(tok, prompts, branch):
    from srd.engine.ref import RefPolicy

    s = make_sample()
    tr = trace_two_rounds(s)
    pol = RefPolicy(_tiny(), tok, prompts, branch=branch)
    b = pol.force(Episode(s), tr)
    _same(b.seq(), build_seq(s, tr, tok, prompts, view="chain", branch=branch))


@pytest.mark.parametrize("branch", ["sealedread", "nomask", "daft"])
def test_generated_trace_layout_equals_trainer(tok, prompts, branch):
    """Sampling with the reference engine, then rebuilding the trace from its text: identical tokens and masks."""
    from srd.engine.ref import RefPolicy

    proto = ProtoConfig(r_max=2, caps={"reason": 12, "query": 6, "recheck": 8, "answer": 4})
    pol = RefPolicy(_tiny(), tok, prompts, branch=branch, proto=proto, temperature=1.0)
    s = make_sample(n=6, gold_pos=(1, 4))
    eps = []
    orig_release = pol.release
    pol.release = lambda ep: eps.append(ep)          # keep the state to compare
    run_episodes(pol, [s], n=2, cfg=DriverConfig(proto=proto, lenient=True))
    assert len(eps) == 2
    for ep in eps:
        assert ep.steps[-1].tag == "answer", ep.error
        got = ep.state["b"].seq()
        _same(got, build_seq(s, ep.steps, tok, prompts, view="chain", branch=branch))
        orig_release(ep)


def test_reference_masked_view_with_oracle(tok, prompts):
    from srd.engine.ref import RefPolicy

    proto = ProtoConfig(caps={"reason": 8, "query": 6, "recheck": 8, "answer": 4})
    for frame in ("full", "compact"):
        pol = RefPolicy(_tiny(), tok, prompts, view="masked", frame=frame, proto=proto)
        eps = run_episodes(pol, [make_sample()], n=1, cfg=DriverConfig(proto=proto, locator="oracle", lenient=True))
        assert eps[0].steps[-1].tag == "answer"
