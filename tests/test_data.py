"""Records in the official formats -> QA items -> samples, and the data checks, on synthetic data."""
from __future__ import annotations

import random

import pytest

from srd.data.build import BuildConfig, PassagePool, build_sample, entity_names, item_of
from srd.data.checks import (auc, check_gold_mapping, check_leaks, check_scripted, check_structure, check_surface)
from srd.data.chunk import chunk_passage, merge_short
from srd.data.fiction import fictionalize
from srd.data.ids import assign_ids, forbidden_ids
from srd.data.leak import leaks
from srd.data.qa import Passage, answer_kind, from_record
from srd.data.synth import enumeration_items, nocontext_question
from srd.text import approx_tokens, contains, standalone_numbers

NATIONS = ["French", "German", "Italian", "Spanish", "Polish", "Dutch"]


def wiki_record(i: int) -> dict:
    film, person, nation = f"Film Alpha {i}", f"Director Person {i}", NATIONS[i % len(NATIONS)]
    ctx = [[film, [f"{film} is a {1950 + i} drama film directed by {person}.", " It was shot on location."]],
           [person, [f"{person} is a {nation} film director.", f" He was active since {1940 + i}."]]]
    ctx += [[f"Filler Topic {i}-{j}", [f"Filler Topic {i}-{j} is a village with {100 + j} inhabitants.",
                                       " It is known for its market."]] for j in range(6)]
    return {"_id": f"w{i}", "question": f"What is the nationality of the director of {film}?", "answer": nation,
            "type": "compositional", "supporting_facts": [[film, 0], [person, 0]], "context": ctx,
            "evidences": [[film, "director", person], [person, "country of citizenship", nation]]}


def items(n=40):
    return [from_record(wiki_record(i), "2wikimultihopqa") for i in range(n)]


def test_parse_wiki_record():
    it = items(1)[0]
    assert [p.gold for p in it.passages][:2] == [True, True]
    assert it.hops[0].titles == ["Film Alpha 0"] and it.hops[0].bridge == "Director Person 0"
    assert it.answer_kind == "open"


def test_hotpot_bridge_order():
    r = {"_id": "h1", "question": "Where was the director of X born?", "answer": "Porto", "type": "bridge",
         "supporting_facts": [["X", 0], ["Mara Elling", 0]],
         "context": [["Mara Elling", ["Mara Elling was born in Porto."]], ["X", ["X is a film by Mara Elling."]]]}
    it = from_record(r, "hotpotqa")
    assert [h.titles[0] for h in it.hops] == ["X", "Mara Elling"]


def test_musique_record():
    r = {"id": "2hop__1_2", "question": "q", "answer": "B", "answerable": True,
         "paragraphs": [{"idx": 0, "title": "T0", "paragraph_text": "T0 links to A.", "is_supporting": True},
                        {"idx": 1, "title": "T1", "paragraph_text": "A is B.", "is_supporting": True},
                        {"idx": 2, "title": "T2", "paragraph_text": "noise", "is_supporting": False}],
         "question_decomposition": [{"question": "x", "answer": "A", "paragraph_support_idx": 0},
                                    {"question": "#1 y", "answer": "B", "paragraph_support_idx": 1}]}
    it = from_record(r, "musique")
    assert [h.bridge for h in it.hops] == ["A", "B"] and it.qtype == "2hop"
    assert from_record({**r, "answerable": False}, "musique") is None


def test_answer_kind():
    assert answer_kind("Is X older than Y?", "yes", []) == "closed"
    assert answer_kind("Who was born first, X or Y?", "X", ["X", "Y"]) == "closed"
    assert answer_kind("How many?", "3", []) == "closed"
    assert answer_kind("Where?", "Porto", []) == "open"


def test_chunking_limits_and_gold():
    p = Passage("T", "", [f"Sentence number {i} has some words in it." for i in range(60)], [37], True)
    cs = merge_short(chunk_passage(p, approx_tokens, 64), approx_tokens, 8, 64)
    assert all(approx_tokens(c.text) <= 64 + 16 for c in cs)
    gold = [c for c in cs if c.gold]
    assert len(gold) == 1 and "number 37" in gold[0].text


def test_ids_avoid_numbers():
    forb = forbidden_ids(["born 1961, 23 films", "page 7"])
    assert {1961, 23, 7} <= forb
    ids = assign_ids(500, random.Random(0), forb)
    assert len(set(ids)) == 500 and not set(ids) & forb


def test_leak_filter():
    it = items(1)[0]
    assert leaks("He is a famous French singer.", it)          # contains the answer
    assert leaks("Film Alpha 0 was made by Director Person 0.", it)   # restates a bridge fact
    assert not leaks("A village with a market.", it)
    r = {"_id": "c", "question": "Who was born first, Ann Lee or Bo Kim?", "answer": "Ann Lee", "type": "comparison",
         "supporting_facts": [["Ann Lee", 0], ["Bo Kim", 0]],
         "context": [["Ann Lee", ["Ann Lee (born 1950)."]], ["Bo Kim", ["Bo Kim (born 1960)."]]],
         "evidences": [["Ann Lee", "date of birth", "1950"], ["Bo Kim", "date of birth", "1960"]]}
    c = from_record(r, "2wikimultihopqa")
    assert c.answer_kind == "closed"
    assert leaks("Ann Lee, born 1950, was a singer.", c)
    assert not leaks("Ann Lee is mentioned here without the year.", c)


@pytest.fixture(scope="module")
def built():
    its = items(40)
    pool = PassagePool(its)
    rng = random.Random(0)
    return [build_sample(it, pool, rng, target=1500, cfg=BuildConfig(max_seg_tokens=64, hard_distractors=4))
            for it in its[:20]]


def test_built_samples(built):
    for s in built:
        assert 1300 < s.meta["tokens"] <= 1500 + 64 + 40       # the budget counts labels and titles too
        assert len(s.gold) >= 2
        assert [h.sids for h in s.hops] and all(h.sids for h in s.hops)
        ids = [g.sid for g in s.segments]
        assert len(set(ids)) == len(ids)
    positions = [[g.sid for g in s.segments].index(s.gold[0]) / len(s.segments) for s in built]
    assert max(positions) - min(positions) > 0.3            # gold at random positions


def test_checks_pass_on_built(built):
    assert check_structure(built)["pass"]
    assert check_gold_mapping(built)["pass"]
    assert check_leaks(built)["pass"]
    assert check_scripted(built)["pass"]


def test_surface_check_detects_a_planted_shortcut(built):
    import copy

    res = check_surface(built)
    assert 0.0 <= res["auc"] <= 1.0
    bad = copy.deepcopy(built)
    for s in bad:
        for g in s.segments:
            if g.gold:
                g.section = "Evidence"                      # a planted surface cue
    assert check_surface(bad)["auc"] > 0.9


def test_auc():
    import numpy as np

    assert auc(np.array([3., 2., 1.]), np.array([1, 0, 0])) == 1.0
    assert auc(np.array([1., 1.]), np.array([1, 0])) == 0.5


def test_fictional_substitution(built):
    s = built[0]
    it = items(1)[0]
    f = fictionalize(s, entity_names(it), random.Random(1))
    m = f.meta["fiction_map"]
    assert "Director Person 0" in m and f.answers[0] != s.answers[0]
    for g in f.segments:
        assert "Director Person 0" not in g.text and "Film Alpha 0" not in g.title
    assert contains(" ".join(g.text for g in f.segments if g.gold), f.answers[0])
    assert check_scripted([f])["pass"]


def test_synthetic_types():
    rng = random.Random(0)
    q = nocontext_question(rng, 0)
    assert q.passages == [] and q.answers[0]
    ens = enumeration_items(items(40), rng, 5)
    assert ens, "groups of 3+ subjects sharing a value"
    for e in ens:
        assert 3 <= len(e.passages) <= 6 and e.leak_terms


def test_nocontext_sample(built):
    its = items(40)
    pool = PassagePool(its)
    s = build_sample(nocontext_question(random.Random(0), 0), pool, random.Random(0), target=800)
    assert s.qtype == "nocontext" and s.gold == [] and s.segments
    assert check_structure([s])["pass"]


def test_item_of_roundtrip(built):
    it = item_of(built[0])
    assert it.gold_titles and it.hops and it.triples


def test_standalone_numbers():
    assert standalone_numbers("in 1961, 3 films; S_871 and v2.5") == {1961, 3}
