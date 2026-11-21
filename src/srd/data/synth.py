"""Synthetic question types of table 4.

    nocontext    type 5 (|G| = 0): self-contained arithmetic / logic / spelling questions; teach answering without
                 opening the context. No questions about real entities.
    enumeration  type 4: count / list questions built from 3-6 2Wiki segments sharing one attribute (e.g. the films of
                 one director); G = every segment needed; distractors may not mention the shared value.
(The comparison and bridge-comparison questions of 2Wiki are used as they are, also as type 4.)
"""
from __future__ import annotations

import random
from collections import defaultdict

from srd.data.qa import Passage, QAItem

NAMES = ["Alice", "Bruno", "Chiara", "Dmitri", "Elena", "Farid", "Greta", "Hiro", "Imani", "Jonas", "Kalinda", "Lars"]
ITEMS = ["apples", "books", "coins", "marbles", "pencils", "stamps", "tickets", "shells"]


def _arith(rng: random.Random) -> tuple[str, str]:
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    k = rng.randrange(6)
    if k == 0:
        return f"What is {a} plus {b}?", str(a + b)
    if k == 1:
        a, b = max(a, b), min(a, b)
        return f"What is {a} minus {b}?", str(a - b)
    if k == 2:
        a, b = rng.randint(3, 19), rng.randint(3, 19)
        return f"What is {a} times {b}?", str(a * b)
    if k == 3:
        h = rng.randint(2, 9)
        return f"How many minutes are there in {h} hours?", str(60 * h)
    if k == 4:
        n, x, y = rng.choice(NAMES), rng.randint(10, 40), rng.randint(2, 9)
        it = rng.choice(ITEMS)
        return f"{n} has {x} {it} and gives away {y}. How many {it} does {n} have left?", str(x - y)
    s, t = rng.randint(40, 120), rng.randint(2, 6)
    return f"A train travels at {s} km per hour for {t} hours. How many kilometres does it travel?", str(s * t)


def _logic(rng: random.Random) -> tuple[str, str]:
    a, b, c = rng.sample(NAMES, 3)
    k = rng.randrange(3)
    if k == 0:
        return f"{a} is taller than {b}, and {b} is taller than {c}. Who is the shortest?", c
    if k == 1:
        return f"{a} is older than {b}, and {c} is older than {a}. Who is the oldest?", c
    w = rng.choice(["harbor", "lantern", "meadow", "orchestra", "quartz", "violet"])
    return f"How many letters are in the word '{w}'?", str(len(w))


def nocontext_question(rng: random.Random, i: int) -> QAItem:
    q, a = (_arith if rng.random() < 0.7 else _logic)(rng)
    return QAItem(qid=f"nc{i}", dataset="synthetic", question=q, answers=[a], passages=[], qtype="nocontext",
                  answer_kind="closed" if a.isdigit() and int(a) <= 10 else "open")


REL_TEMPLATES = {
    "director": ("films", "were directed by {o}"),
    "performer": ("works", "were performed by {o}"),
    "composer": ("works", "had music composed by {o}"),
    "producer": ("works", "were produced by {o}"),
    "publisher": ("works", "were published by {o}"),
    "country of origin": ("works", "come from {o}"),
    "country of citizenship": ("people", "are citizens of {o}"),
    "place of birth": ("people", "were born in {o}"),
}


def enumeration_items(items: list[QAItem], rng: random.Random, n: int, kind: str = "mixed") -> list[QAItem]:
    """Count / list questions over groups of 3-6 subjects that share (relation, value) in 2Wiki evidences."""
    groups: dict[tuple[str, str], dict[str, Passage]] = defaultdict(dict)
    for it in items:
        if it.dataset != "2wikimultihopqa":
            continue
        by_title = {p.title: p for p in it.passages if p.gold}
        for s, r, o in it.triples:
            rel = r.replace("_", " ")
            if rel in REL_TEMPLATES and s in by_title:
                groups[(rel, o)][s] = by_title[s]
    keys = [k for k, v in groups.items() if len(v) >= 3]
    rng.shuffle(keys)
    out = []
    for j, (rel, o) in enumerate(keys[:n]):
        subs = list(groups[(rel, o)].items())
        rng.shuffle(subs)
        subs = subs[: rng.randint(3, min(6, len(subs)))]
        noun, pred = REL_TEMPLATES[rel]
        passages = [Passage(p.title, p.text, p.sentences, p.support, True) for _, p in subs]
        as_list = kind == "list" or (kind == "mixed" and rng.random() < 0.5)
        if as_list:
            q = f"Which of the {noun} described in the documents {pred.format(o=o)}? List all of them."
            ans, ak = [s for s, _ in subs], "list"
        else:
            q = f"How many of the {noun} described in the documents {pred.format(o=o)}?"
            ans, ak = [str(len(subs))], "closed"
        out.append(QAItem(qid=f"enum{j}", dataset="2wikimultihopqa", question=q, answers=ans, passages=passages,
                          qtype="enumeration", answer_kind=ak, leak_terms=[o]))
    return out
