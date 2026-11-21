"""Mask probes on a tiny model, and the benchmark converters on records in the official layouts."""
from __future__ import annotations

import pytest

from conftest import trace_two_rounds
from srd.data.benchmarks import build_docs_samples, disambiguate, qasper_docs, quality_docs
from srd.text import approx_tokens

torch = pytest.importorskip("torch")


def _tiny():
    from srd.model.qwen3 import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    m = Qwen3ForCausalLM(Qwen3Config.tiny()).eval()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return m


def test_invariance_probe(sample, tok, prompts):
    from srd.eval.probes import invariance

    m = _tiny()
    tr = trace_two_rounds(sample)
    sr = invariance(m, sample, tr, tok, prompts, "sealedread")
    nm = invariance(m, sample, tr, tok, prompts, "nomask")
    assert sr["kl_max"] < 1e-9 and nm["kl_max"] > 1e-6        # exact zero vs a measurable leak
    assert sr["positions"] > 50


def test_dependence_probe(sample, tok, prompts):
    from srd.eval.probes import dependence

    m = _tiny()
    r = dependence(m, sample, trace_two_rounds(sample), tok, prompts, "Lisbon")
    assert r is not None and r["drop"] != 0.0


def qasper_record(i):
    paras = [[f"Paragraph {j} of paper {i} says thing {j}." for j in range(6)]]
    return {"id": f"p{i}", "title": f"Paper {i}", "abstract": "We study things.",
            "full_text": {"section_name": ["Intro"], "paragraphs": paras},
            "qas": {"question": ["What do the authors say?"],
                    "answers": [{"answer": [{"unanswerable": False, "yes_no": None, "extractive_spans": ["thing 3"],
                                             "free_form_answer": "", "evidence": [paras[0][3]]}]}]}}


def test_qasper_conversion():
    docs = qasper_docs([qasper_record(i) for i in range(30)])
    ss = build_docs_samples(docs, "qasper", [600], approx_tokens)
    assert ss and all(s.gold for s in ss)
    s = ss[0]
    assert s.question.startswith("In the paper 'Paper 0'") and s.answers == ["thing 3"]
    assert len({g.doc for g in s.segments}) > 1                # padded with other papers


def test_quality_conversion():
    rec = {"article_id": "a1", "title": "Story", "article": "<p>Once upon a time.</p>\n\n<p>The end.</p>",
           "questions": [{"question": "How does it end?", "options": ["Badly", "The end", "No", "Yes"],
                          "gold_label": 2}]}
    docs = quality_docs([rec])
    assert docs[0]["qas"][0]["answers"][0] == "B"
    assert disambiguate("What do they propose?", "X") == "In the paper 'X', what do they propose?"
