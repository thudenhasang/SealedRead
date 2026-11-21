"""Multi-hop QA records -> QAItem: passages with sentence-level support, ordered hops, answer kind.

Input records are the official HotpotQA / 2WikiMultihopQA / MuSiQue records as stored in ragproj2's shards
(`<dataset>_s<k>.json`, the same files keychain-dv reads; scripts/fetch_data.py).

    hops         ordered reasoning steps of bridge questions (hop-skip rate, section 6.2): MuSiQue's decomposition,
                 2Wiki's evidence triples (compositional / inference types), HotpotQA bridge questions (the gold
                 passage that names the other one's title comes first). Comparisons have no order.
    answer_kind  closed: yes/no, one of the two compared entities, a small count (<= 10); else open (section 6.2)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from srd.text import contains, is_weak_span, normalize

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")


@dataclass
class Passage:
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)   # empty: the passage is one unit
    support: list[int] = field(default_factory=list)     # supporting sentence indices (gold passages)
    gold: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return (self.title, self.text)


@dataclass
class HopSpec:
    titles: list[str]
    bridge: str


@dataclass
class QAItem:
    qid: str
    dataset: str
    question: str
    answers: list[str]
    passages: list[Passage]
    qtype: str = ""
    hops: list[HopSpec] = field(default_factory=list)
    units: list[str] = field(default_factory=list)       # intermediate / final spans (leak filter)
    triples: list[list[str]] = field(default_factory=list)   # 2Wiki evidences (s, r, o)
    answer_kind: str = "open"
    leak_terms: list[str] = field(default_factory=list)      # extra strings no distractor may contain

    @property
    def gold_passages(self) -> list[Passage]:
        return [p for p in self.passages if p.gold]

    @property
    def gold_titles(self) -> list[str]:
        return list(dict.fromkeys(p.title for p in self.passages if p.gold))

    @property
    def ordered(self) -> bool:
        return len(self.hops) >= 2

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "QAItem":
        return QAItem(qid=d["qid"], dataset=d["dataset"], question=d["question"], answers=list(d["answers"]),
                      passages=[Passage(**p) for p in d["passages"]], qtype=d.get("qtype", ""),
                      hops=[HopSpec(**h) for h in d.get("hops", [])], units=list(d.get("units", [])),
                      triples=[list(t) for t in d.get("triples", [])], answer_kind=d.get("answer_kind", "open"),
                      leak_terms=list(d.get("leak_terms", [])))


def base_dataset(name: str) -> str:
    return re.sub(r"_s\d+$", "", Path(name).stem.replace("_corpus", ""))


def answer_kind(question: str, answer: str, entities: list[str]) -> str:
    a = normalize(answer)
    if a in ("yes", "no"):
        return "closed"
    if a.isdigit() and int(a) <= 10:
        return "closed"
    if any(a and a == normalize(e) for e in entities) or (entities and contains(question, answer)):
        return "closed"
    return "open"


def _hotpot_like(r: dict, dataset: str) -> QAItem:
    sep = "" if dataset == "hotpotqa" else " "
    sup: dict[str, set[int]] = {}
    for t, i in r["supporting_facts"]:
        sup.setdefault(t, set()).add(int(i))
    passages = []
    for t, sents in r["context"]:
        sents = [s for s in sents]
        support = sorted(i for i in sup.get(t, ()) if i < len(sents))
        passages.append(Passage(t, sep.join(sents).strip(), [s.strip() for s in sents], support, bool(support)))
    gold = [p for p in passages if p.gold]
    triples = [list(e) for e in r.get("evidences", [])]
    qtype = r.get("type", "")
    hops: list[HopSpec] = []
    units = [r["answer"]]
    if dataset == "2wikimultihopqa":
        units += [o for _, _, o in triples]
        if qtype in ("compositional", "inference") and triples:
            titles = {p.title for p in gold}
            hops = [HopSpec([s] if s in titles else [], o) for s, _, o in triples]
            if any(not h.titles for h in hops):
                hops = []
    elif dataset == "hotpotqa" and qtype == "bridge" and len(gold) == 2:
        a, b = gold
        a_names_b, b_names_a = contains(a.text, b.title), contains(b.text, a.title)
        if a_names_b != b_names_a:
            first, second = (a, b) if a_names_b else (b, a)
            hops = [HopSpec([first.title], second.title), HopSpec([second.title], r["answer"])]
            units.append(second.title)
    entities = [p.title for p in gold] if qtype in ("comparison", "bridge_comparison") else []
    return QAItem(qid=str(r["_id"]), dataset=dataset, question=r["question"], answers=[r["answer"]],
                  passages=passages, qtype=qtype, hops=hops, units=list(dict.fromkeys(units)), triples=triples,
                  answer_kind=answer_kind(r["question"], r["answer"], entities))


def _musique(r: dict) -> QAItem:
    paras = r["paragraphs"]
    by_idx = {p["idx"]: i for i, p in enumerate(paras)}
    passages = [Passage(p["title"], p["paragraph_text"], gold=bool(p.get("is_supporting"))) for p in paras]
    hops, units, answers_so_far = [], [], []
    for step in r.get("question_decomposition", []):
        j = by_idx.get(step.get("paragraph_support_idx"))
        hops.append(HopSpec([passages[j].title] if j is not None else [], step["answer"]))
        answers_so_far.append(step["answer"])
        units.append(step["answer"])
    if any(not h.titles for h in hops):
        hops = []
    m = re.match(r"(\d)hop", str(r.get("id", "")))
    return QAItem(qid=str(r["id"]), dataset="musique", question=r["question"],
                  answers=[r["answer"]] + list(r.get("answer_aliases", [])), passages=passages,
                  qtype=f"{m.group(1)}hop" if m else "", hops=hops, units=list(dict.fromkeys(units)),
                  answer_kind=answer_kind(r["question"], r["answer"], []))


def from_record(r: dict, dataset: str) -> QAItem | None:
    dataset = base_dataset(dataset)
    if dataset == "musique":
        if not r.get("answerable", True):
            return None
        item = _musique(r)
    elif dataset in ("hotpotqa", "2wikimultihopqa"):
        item = _hotpot_like(r, dataset)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if not item.gold_passages or not normalize(item.answers[0]):
        return None
    return item


def load_shard(path: str | Path) -> list[QAItem]:
    path = Path(path)
    dataset = base_dataset(path.name)
    records = json.loads(path.read_text(encoding="utf-8"))
    return [it for it in (from_record(r, dataset) for r in records) if it is not None]


def is_weak(span: str) -> bool:
    return is_weak_span(span)
