"""One training / evaluation sample: a question over a long context C made of ID-labelled segments."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Segment:
    sid: int                 # the random ID shown as [[S_sid]] (1..9999, unique in the sample)
    title: str               # document title
    section: str             # section of the document ("Introduction" for lead paragraphs)
    text: str
    doc: str = ""            # source document key (segments of one document share it)
    gold: bool = False       # in G: holds at least one supporting fact


@dataclass
class Hop:
    """One reasoning step of a bridge question (for the hop-skip rate, section 6.2)."""
    sids: list[int]          # segments that support the step
    bridge: str = ""         # what the step reveals (its answer); step j+1 should not be located before this appears


@dataclass
class Sample:
    qid: str
    dataset: str
    qtype: str               # multihop | paraphrase | fictional | multiseg | nocontext | benchmark
    question: str
    answers: list[str]       # gold answer first, then aliases; for answer_kind "list": the gold items
    answer_kind: str         # open | closed | list
    segments: list[Segment]  # in the order of C
    hops: list[Hop] = field(default_factory=list)
    cluster: str = ""        # bootstrap cluster: the root question (all paraphrases / lengths of one question)
    length: int = 0          # target length of C in tokens (bucket: 16K-24K, 32K, 64K, 128K)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ lookups
    @property
    def gold(self) -> list[int]:
        return [s.sid for s in self.segments if s.gold]

    def order(self) -> dict[int, int]:
        """sid -> position of the segment in C."""
        return {s.sid: i for i, s in enumerate(self.segments)}

    def by_sid(self, sid: int) -> Segment:
        for s in self.segments:
            if s.sid == sid:
                return s
        raise KeyError(sid)

    def ids(self) -> set[int]:
        return {s.sid for s in self.segments}

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Sample":
        return Sample(
            qid=d["qid"], dataset=d["dataset"], qtype=d["qtype"], question=d["question"], answers=list(d["answers"]),
            answer_kind=d.get("answer_kind", "open"), segments=[Segment(**s) for s in d["segments"]],
            hops=[Hop(**h) for h in d.get("hops", [])], cluster=d.get("cluster", d["qid"]),
            length=d.get("length", 0), meta=d.get("meta", {}))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_samples(path: str | Path, limit: int | None = None) -> list[Sample]:
    out = []
    for d in read_jsonl(path):
        out.append(Sample.from_dict(d))
        if limit and len(out) >= limit:
            break
    return out
