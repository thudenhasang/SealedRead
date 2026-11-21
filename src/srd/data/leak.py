"""Answer-leak filter for distractors (section 4.2), by answer type.

(a) The answer is an entity / value that is not in the question: drop distractors that contain the answer (after
    normalization) or restate a bridge fact (a hop's subject together with what the hop reveals).
(b) Yes/no, or a comparison whose answer is an entity named in the question: filtering on the answer string would
    remove every hard distractor (the string is everywhere); instead drop distractors that state the compared
    attribute of the compared entities (2Wiki evidences), or, without evidences (HotpotQA), any distractor that
    names a compared entity.
Goal in both cases: the evidence gate must not punish a model that answered from a distractor that was evidence.
"""
from __future__ import annotations

from srd.data.qa import QAItem
from srd.text import contains, normalize


def choice_answer(item: QAItem) -> bool:
    """Case (b): yes/no, or the answer is one of the entities the question names."""
    a = normalize(item.answers[0])
    if a in ("yes", "no"):
        return True
    if item.answer_kind != "closed" or a.isdigit():
        return False
    return a in {normalize(t) for t in item.gold_titles} or contains(item.question, item.answers[0])


def leaks(text: str, item: QAItem, title: str = "") -> bool:
    full = f"{title}. {text}" if title else text
    if any(contains(full, t) for t in item.leak_terms):
        return True
    if choice_answer(item):
        if item.triples:
            return any(contains(full, s) and contains(full, o) for s, _, o in item.triples)
        return any(contains(full, e) for e in item.gold_titles)
    for a in item.answers:
        if len(normalize(a)) > 2 and contains(full, a):
            return True
    for h in item.hops[:-1]:                      # bridge facts: subject + revealed entity together
        for t in h.titles:
            if h.bridge and contains(full, t) and contains(full, h.bridge):
                return True
    return False
