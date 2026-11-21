"""QA items -> samples: gold + own distractors + BM25 hard distractors + filler up to a token budget, chunked into
ID-labelled segments (sections 3.1 and 4).

Recipe of one sample (every type, every length):
    1. the item's own passages: gold ones always; its own distractors if they pass the leak filter
    2. hard distractors: BM25 over the pool with the question and the gold titles, leak-filtered, other titles
    3. filler from the pool, leak-filtered, until the budget of C is reached
    4. documents in a random order (gold at random positions), each chunked (<= 256 tokens, sentence boundaries,
       short chunks merged within the document)
    5. random IDs that avoid the numbers of the text; G = chunks that hold a supporting sentence; hops -> IDs
Train samples get a random length in [lo, hi]; test samples are built at each fixed length from the same question.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterator

from srd.data.bm25 import BM25
from srd.data.chunk import Chunk, chunk_passage, merge_short
from srd.data.ids import assign_ids, forbidden_ids
from srd.data.leak import leaks
from srd.data.qa import Passage, QAItem
from srd.data.sample import Hop, Sample, Segment
from srd.text import approx_tokens

TokenCounter = Callable[[str], int]
SEG_OVERHEAD = 16          # labels '[[S_k]] (... / ...) ... [[/S_k]]' and separators, in tokens


@dataclass
class BuildConfig:
    max_seg_tokens: int = 256
    min_seg_tokens: int = 32
    hard_distractors: int = 8
    section: str = "Introduction"


class PassagePool:
    """Deduplicated passages of many items (distractors and filler), with a lazily built BM25 index."""

    def __init__(self, items: list[QAItem]):
        self.passages: list[Passage] = []
        seen: set[tuple[str, str]] = set()
        for it in items:
            for p in it.passages:
                if p.key not in seen:
                    seen.add(p.key)
                    self.passages.append(Passage(p.title, p.text, p.sentences, [], False))
        self._bm25: BM25 | None = None

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([f"{p.title}. {p.text}" for p in self.passages])
        return self._bm25

    def hard(self, query: str, k: int, ok: Callable[[Passage], bool], exclude_titles: set[str]) -> list[Passage]:
        out = []
        for i in self.bm25.top(query, 20 * k + 20):
            p = self.passages[i]
            if p.title in exclude_titles or not ok(p):
                continue
            out.append(p)
            exclude_titles = exclude_titles | {p.title}
            if len(out) >= k:
                break
        return out

    def filler(self, rng: random.Random) -> Iterator[Passage]:
        order = list(range(len(self.passages)))
        rng.shuffle(order)
        for i in order:
            yield self.passages[i]


def passage_cost(p: Passage, count: TokenCounter) -> int:
    return count(p.text) + count(p.title) + SEG_OVERHEAD


def build_sample(item: QAItem, pool: PassagePool, rng: random.Random, target: int, count: TokenCounter = approx_tokens,
                 cfg: BuildConfig | None = None, qtype: str | None = None, cluster: str | None = None,
                 qid: str | None = None) -> Sample:
    cfg = cfg or BuildConfig()

    def ok(p: Passage) -> bool:
        return not leaks(p.text, item, p.title)

    gold = [p for p in item.passages if p.gold]
    own = [p for p in item.passages if not p.gold and ok(p)]
    titles = {p.title for p in gold}
    docs: list[Passage] = []
    keys: set[tuple[str, str]] = set()
    for p in gold + own:
        if p.key not in keys and (p.gold or p.title not in titles):
            docs.append(p)
            keys.add(p.key)
            titles.add(p.title)
    if gold and cfg.hard_distractors:
        q = item.question + " " + " ".join(item.gold_titles)
        for p in pool.hard(q, cfg.hard_distractors, ok, set(titles)):
            if p.key not in keys:
                docs.append(p)
                keys.add(p.key)
                titles.add(p.title)
    used = sum(passage_cost(p, count) for p in docs)
    for p in pool.filler(rng):
        if used >= target:
            break
        if p.key in keys or p.title in titles:
            continue
        c = passage_cost(p, count)
        if used + c > target + cfg.max_seg_tokens:
            continue
        if not ok(p):
            continue
        docs.append(p)
        keys.add(p.key)
        titles.add(p.title)
        used += c
    rng.shuffle(docs)
    spans = [h.bridge for h in item.hops] + list(item.units) + list(item.answers)
    chunks: list[Chunk] = []
    for p in docs:
        cs = chunk_passage(p, count, cfg.max_seg_tokens, doc=p.title, section=cfg.section, gold_spans=spans)
        chunks.extend(merge_short(cs, count, cfg.min_seg_tokens, cfg.max_seg_tokens))
    ids = assign_ids(len(chunks), rng, forbidden_ids([c.text + " " + c.title for c in chunks] + [item.question]))
    segments = [Segment(sid, c.title, c.section, c.text, c.doc, c.gold) for sid, c in zip(ids, chunks)]
    hops: list[Hop] = []
    for h in item.hops:
        sids = [s.sid for s in segments if s.gold and s.title in h.titles]
        if not sids:
            hops = []
            break
        hops.append(Hop(sids, h.bridge))
    return Sample(qid=qid or f"{item.dataset}-{item.qid}", dataset=item.dataset,
                  qtype=qtype or ("nocontext" if not gold else "multihop"), question=item.question,
                  answers=list(item.answers), answer_kind=item.answer_kind, segments=segments, hops=hops,
                  cluster=cluster or f"{item.dataset}-{item.qid}", length=target,
                  meta={"src_qtype": item.qtype, "tokens": used, "triples": item.triples,
                        "leak_terms": item.leak_terms, "units": item.units,
                        "hop_titles": [h.titles for h in item.hops]})


def item_of(s: Sample) -> QAItem:
    """Enough of the source item to re-run the leak filter on a built sample (data check 4)."""
    from srd.data.qa import HopSpec

    gold_titles = list(dict.fromkeys(g.title for g in s.segments if g.gold))
    passages = [Passage(t, "", gold=True) for t in gold_titles]
    hops = [HopSpec(list(t), h.bridge) for t, h in zip(s.meta.get("hop_titles", []), s.hops)]
    return QAItem(qid=s.qid, dataset=s.dataset, question=s.question, answers=list(s.answers), passages=passages,
                  qtype=s.meta.get("src_qtype", ""), hops=hops, units=list(s.meta.get("units", [])),
                  triples=[list(t) for t in s.meta.get("triples", [])], answer_kind=s.answer_kind,
                  leak_terms=list(s.meta.get("leak_terms", [])))


def entity_names(item: QAItem) -> list[str]:
    """Names to rename for the fictional set: gold titles, hop bridges, 2Wiki subjects/objects, name-like answers."""
    out = list(item.gold_titles) + [h.bridge for h in item.hops] + list(item.answers)
    for s, _, o in item.triples:
        out += [s, o]
    return out
