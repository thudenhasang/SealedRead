"""Paraphrased questions (type 2 of table 4): a strong model rewrites the question to reduce word overlap with the
gold segments on verbs and relations while keeping entity names; the rewrite is kept only if a model answering
from the gold segments gives the same answer (scripts/paraphrase.py runs both with vLLM). This teaches tolerance to
surface variation, not NoLiMa-style latent association (NoLiMa stays out-of-distribution evaluation)."""
from __future__ import annotations

from srd.data.sample import Sample
from srd.text import content_words, contains

REWRITE = """\
Rewrite the question so that it asks exactly the same thing with different wording: change verbs, relation words \
and phrasing so that it shares as few words as possible with the facts below, but keep every name exactly as \
written. Output only the rewritten question.

Facts:
{facts}

Question: {question}
Rewritten question:"""

ANSWER = """\
Answer the question using only the facts below. Output only the answer, as short as possible.

Facts:
{facts}

Question: {question}
Answer:"""


def gold_facts(s: Sample) -> str:
    return "\n".join(f"- {g.title}: {g.text}" for g in s.segments if g.gold)


def overlap(question: str, s: Sample) -> float:
    """Share of the question's content words that occur in the gold segments."""
    q = content_words(question)
    if not q:
        return 0.0
    gold = " ".join(g.title + " " + g.text for g in s.segments if g.gold)
    return sum(contains(gold, w) for w in q) / len(q)


def keeps_names(orig: str, new: str, names: list[str]) -> bool:
    return all(contains(new, n) for n in names if contains(orig, n))


def apply(s: Sample, new_question: str) -> Sample:
    d = s.to_dict()
    d["meta"] = {**d.get("meta", {}), "orig_question": s.question, "overlap_before": overlap(s.question, s),
                 "overlap_after": overlap(new_question, s)}
    d["question"] = new_question.strip()
    d["qtype"] = "paraphrase"
    d["qid"] = s.qid + "-para"
    return Sample.from_dict(d)
