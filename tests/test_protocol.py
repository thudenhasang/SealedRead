from __future__ import annotations

import pytest

from conftest import make_sample, trace_two_rounds
from srd.grammar import LocateTrie, locate_regex
from srd.protocol import (ProtoConfig, ProtocolError, Step, force_alpha_answer, next_tag, parse_alpha, parse_beta,
                          parse_ids, validate)
from srd.tok import ByteTok


def _val(s, steps, cfg=None):
    return validate(steps, s.ids(), s.order(), cfg or ProtoConfig())


def test_example_trace_is_valid(sample, trace):
    assert _val(sample, trace) == []


def test_state_machine_sequence(sample, trace):
    cfg = ProtoConfig()
    want = [st.tag for st in trace]
    got = [next_tag(trace[:i], cfg) for i in range(len(trace))]
    assert got == want
    assert next_tag(trace, cfg) is None


def test_alpha_beta_parsing():
    assert parse_alpha("x alpha=query") == "query"
    assert parse_alpha("alpha = answer ") == "answer"
    assert parse_alpha("alpha=query and more") is None
    assert parse_beta("ok beta=reject_and_requery") == "reject_and_requery"
    assert force_alpha_answer("so alpha=query") == "so alpha=answer"
    assert force_alpha_answer("no alpha") == "no alpha alpha=answer"
    assert parse_ids("[[S_871]] [[S_2305]]") == [871, 2305]


def test_reject_and_requery_skips_reason(sample):
    g = sample.gold
    steps = [Step("reason", "alpha=query"), Step("query", "x"), Step("locate", ids=[g[0]]),
             Step("knowledge", ids=[g[0]], by="engine"), Step("recheck", "no beta=reject_and_requery")]
    assert next_tag(steps, ProtoConfig()) == "query"


def test_reject_at_rmax_goes_to_reason(sample):
    """Fix D3: at round R_max reject_and_requery may not open a new round."""
    g = sample.gold
    steps = [Step("reason", "alpha=query"), Step("query", "x"), Step("locate", ids=[g[0]]),
             Step("knowledge", ids=[g[0]], by="engine"), Step("recheck", "beta=reject_and_requery")]
    assert next_tag(steps, ProtoConfig(r_max=1)) == "reason"
    bad = steps + [Step("reason", "alpha=query")]
    with pytest.raises(ProtocolError):
        next_tag(bad, ProtoConfig(r_max=1))
    errs = _val(sample, bad + [Step("answer", "x")], ProtoConfig(r_max=1))
    assert errs


@pytest.mark.parametrize("ids,msg", [([], "0 IDs"), ([1, 1], "repeats"), ([99999], "does not exist")])
def test_bad_locate(sample, ids, msg):
    steps = [Step("reason", "alpha=query"), Step("query", "x"), Step("locate", ids=ids),
             Step("knowledge", ids=[], by="engine"), Step("recheck", "beta=valid_to_think"),
             Step("reason", "alpha=answer"), Step("answer", "x")]
    errs = _val(sample, steps)
    assert any(msg in e for e in errs), errs


def test_repasting_is_a_violation(sample, trace):
    t = list(trace)
    t[7] = Step("locate", ids=[trace[2].ids[0]])        # the first round's segment again
    t[8] = Step("knowledge", ids=[trace[2].ids[0]], by="engine")
    assert any("already pasted" in e for e in _val(sample, t))


def test_knowledge_must_be_engine_and_in_c_order(sample, trace):
    t = list(trace)
    t[3] = Step("knowledge", ids=list(reversed(trace[3].ids)), by="engine")
    assert _val(sample, t)


def test_two_answers_invalid(sample, trace):
    assert _val(sample, trace + [Step("answer", "x")])


def test_caps_with_token_count(sample, trace):
    cfg = ProtoConfig(caps={"reason": 5, "query": 64, "recheck": 256, "answer": 64})
    errs = validate(trace, sample.ids(), sample.order(), cfg, token_count=lambda t: len(t))
    assert any("cap" in e for e in errs)


# ---------------------------------------------------------------------- grammar
def _walk(trie, rng, tok):
    out = []
    while not trie.done:
        allowed = trie.allowed()
        t = rng.choice(allowed)
        trie.advance(t)
        out.append(t)
    return tok.decode(out)


def test_locate_trie_only_produces_legal_blocks():
    import random

    tok = ByteTok()
    allowed = [7, 71, 712, 1200, 9999]
    rng = random.Random(0)
    for _ in range(300):
        trie = LocateTrie(tok, allowed, k=3)
        text = _walk(trie, rng, tok)
        assert text.endswith("</locate>")
        ids = parse_ids(text)
        assert 1 <= len(ids) <= 3 and len(set(ids)) == len(ids) and set(ids) <= set(allowed)
        assert text == " ".join(f"[[S_{i}]]" for i in ids) + "</locate>"


def test_locate_trie_respects_k_and_exhaustion():
    tok = ByteTok()
    trie = LocateTrie(tok, [5], k=3)
    for t in tok.encode("[[S_5]]"):
        trie.advance(t)
    assert trie.allowed() == tok.encode("</locate>")[:1]   # the only ID is used: no separator


def test_locate_regex():
    import re

    rx = re.compile(locate_regex([3, 45], 3))
    assert rx.fullmatch("[[S_3]] [[S_45]]</locate>")
    assert not rx.fullmatch("[[S_4]]</locate>")
    assert not rx.fullmatch("[[S_3]] [[S_3]] [[S_3]] [[S_3]]</locate>")


def test_nocontext_sample_has_no_gold():
    s = make_sample(nocontext=True)
    assert s.gold == []
    assert validate([Step("reason", "17+25=42. alpha=answer"), Step("answer", "42")], s.ids(), s.order(),
                    ProtoConfig()) == []


def test_example_trace_roundtrip(sample):
    from srd.protocol import steps_from_dicts, steps_to_dicts

    t = trace_two_rounds(sample)
    t[2].alias = {t[2].ids[0]: 1234}
    assert [s.to_dict() for s in steps_from_dicts(steps_to_dicts(t))] == [s.to_dict() for s in t]
