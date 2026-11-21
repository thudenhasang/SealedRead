"""Retrievers for the RAG baselines: they replace <locate> and see only the query (section 6.1, RQ2).

    BM25Retriever    no dependency; per-sample index over the segments (title + text)
    DenseRetriever   bi-encoder + cross-encoder reranker (sentence-transformers); model names go in the config and
                     are committed before S0 (pre-registration)
"""
from __future__ import annotations

from srd.data.bm25 import BM25
from srd.data.sample import Sample


def _doc(seg) -> str:
    return f"{seg.title}. {seg.text}"


class BM25Retriever:
    def __init__(self):
        self._idx: dict[str, BM25] = {}

    def _index(self, s: Sample) -> BM25:
        key = f"{s.qid}:{s.length}"
        if key not in self._idx:
            self._idx[key] = BM25([_doc(g) for g in s.segments])
        return self._idx[key]

    def top(self, s: Sample, query: str, k: int, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        ex_pos = {i for i, g in enumerate(s.segments) if g.sid in exclude}
        pos = self._index(s).top(query, k, ex_pos)
        if len(pos) < k:                                  # no term overlap: fill in C order
            pos += [i for i in range(len(s.segments)) if i not in ex_pos and i not in pos][:k - len(pos)]
        return [s.segments[i].sid for i in pos]


class DenseRetriever:
    def __init__(self, encoder: str, reranker: str | None = None, first_stage: int = 50, device: str | None = None,
                 query_prefix: str = "", batch_size: int = 64):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.enc = SentenceTransformer(encoder, device=device)
        self.rr = CrossEncoder(reranker, device=device) if reranker else None
        self.first_stage = first_stage
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._emb: dict[str, object] = {}

    def _embed(self, s: Sample):
        key = f"{s.qid}:{s.length}"
        if key not in self._emb:
            self._emb.clear()                             # one sample at a time: contexts are large
            self._emb[key] = self.enc.encode([_doc(g) for g in s.segments], batch_size=self.batch_size,
                                             normalize_embeddings=True, convert_to_tensor=True)
        return self._emb[key]

    def top(self, s: Sample, query: str, k: int, exclude: set[int] | None = None) -> list[int]:
        import torch

        exclude = exclude or set()
        emb = self._embed(s)
        q = self.enc.encode([self.query_prefix + query], normalize_embeddings=True, convert_to_tensor=True)[0]
        sc = emb @ q
        order = torch.argsort(sc, descending=True).tolist()
        cand = [i for i in order if s.segments[i].sid not in exclude][:self.first_stage]
        if self.rr is not None and cand:
            rs = self.rr.predict([(query, _doc(s.segments[i])) for i in cand])
            cand = [i for _, i in sorted(zip(rs, cand), key=lambda x: -x[0])]
        return [s.segments[i].sid for i in cand[:k]]


def make_retriever(cfg: dict | None):
    cfg = dict(cfg or {})
    kind = cfg.pop("kind", "bm25")
    if kind == "bm25":
        return BM25Retriever()
    if kind == "dense":
        return DenseRetriever(**cfg)
    raise ValueError(f"unknown retriever {kind!r}")
