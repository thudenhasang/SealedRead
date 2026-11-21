"""A small BM25 (Okapi) index: hard distractors for the data builder and the sparse retriever of the RAG baseline."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from srd.text import STOPWORDS, tokens


def terms(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: list[Counter] = [Counter(terms(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        self.post: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                df[t] += 1
                self.post[t].append(i)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for t in set(terms(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.post[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def top(self, query: str, k: int, exclude: set[int] | None = None) -> list[int]:
        sc = self.scores(query)
        ranked = sorted(sc.items(), key=lambda x: (-x[1], x[0]))
        return [i for i, _ in ranked if not exclude or i not in exclude][:k]
