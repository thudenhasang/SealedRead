"""Paragraph chunking (section 3.1, item 2): segments of at most `max_tokens`, cut at sentence boundaries; segments
shorter than `min_tokens` are merged with a neighbour of the same document. Evidence and distractors follow the
same rule. A chunk is gold when it holds a supporting sentence (or, without sentence labels, when its passage is gold
and it holds a gold span / is the passage's only chunk)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from srd.data.qa import Passage
from srd.text import contains

_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s for s in (x.strip() for x in _SENT.split(text)) if s]


@dataclass
class Chunk:
    title: str
    text: str
    doc: str
    section: str = "Introduction"
    gold: bool = False
    sent_ids: list[int] = field(default_factory=list)


def _pack(sents: list[str], count: Callable[[str], int], max_tokens: int) -> list[list[int]]:
    groups, cur, size = [], [], 0
    for i, s in enumerate(sents):
        n = count(s) + 1
        if cur and size + n > max_tokens:
            groups.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        groups.append(cur)
    return groups


def _hard_split(text: str, count: Callable[[str], int], max_tokens: int) -> list[str]:
    words, out, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if count(" ".join(cur)) > max_tokens:
            cur.pop()
            if cur:
                out.append(" ".join(cur))
            cur = [w]
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_passage(p: Passage, count: Callable[[str], int], max_tokens: int = 256, doc: str | None = None,
                  section: str = "Introduction", gold_spans: list[str] | None = None) -> list[Chunk]:
    doc = doc or p.title
    sents = p.sentences or split_sentences(p.text)
    if not sents:
        return []
    expanded: list[tuple[str, int]] = []           # (sentence piece, original sentence index)
    for i, s in enumerate(sents):
        pieces = _hard_split(s, count, max_tokens) if count(s) > max_tokens else [s]
        expanded.extend((x, i) for x in pieces)
    groups = _pack([x for x, _ in expanded], count, max_tokens)
    chunks = []
    for g in groups:
        sids = sorted({expanded[i][1] for i in g})
        text = " ".join(expanded[i][0] for i in g)
        chunks.append(Chunk(p.title, text, doc, section, False, sids))
    if p.gold:
        if p.support:
            for c in chunks:
                c.gold = bool(set(c.sent_ids) & set(p.support))
        else:
            spans = [s for s in (gold_spans or []) if s]
            hit = [c for c in chunks if any(contains(c.text, s) for s in spans)]
            for c in (hit or chunks[:1]) if len(chunks) > 1 else chunks:
                c.gold = True
    return chunks


def merge_short(chunks: list[Chunk], count: Callable[[str], int], min_tokens: int = 32,
                max_tokens: int = 256) -> list[Chunk]:
    """Merge chunks under min_tokens into the previous (else next) chunk of the same document."""
    out: list[Chunk] = []
    for c in chunks:
        if out and out[-1].doc == c.doc and (count(c.text) < min_tokens or count(out[-1].text) < min_tokens) \
                and count(out[-1].text) + count(c.text) <= max_tokens + min_tokens:
            prev = out[-1]
            out[-1] = Chunk(prev.title, prev.text + " " + c.text, prev.doc, prev.section, prev.gold or c.gold,
                            prev.sent_ids + c.sent_ids)
        else:
            out.append(c)
    return out
