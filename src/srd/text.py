"""Text helpers shared by the data builder, the rewards and the metrics (SQuAD-style normalization)."""
from __future__ import annotations

import re
import string
from collections import Counter

_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other
our ours ourselves out over own same she should so some such than that the their theirs them themselves then there
these they this those through to too under until up very was we were what when where which while who whom why will
with would you your yours yourself yourselves also either neither whose whether
""".split())


def normalize(s: str) -> str:
    """Lowercase, drop punctuation and articles, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def tokens(s: str) -> list[str]:
    return normalize(s).split()


def content_words(s: str) -> list[str]:
    """Normalized tokens that are not stopwords, in order, without repeats."""
    return list(dict.fromkeys(t for t in tokens(s) if t not in STOPWORDS))


def f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred: str, golds: list[str]) -> float:
    return max((f1(pred, g) for g in golds), default=0.0)


def exact_match(pred: str, golds: list[str]) -> bool:
    return any(normalize(pred) == normalize(g) for g in golds)


def split_list(answer: str) -> list[str]:
    """'A, B and C' / 'A; B' -> ['A', 'B', 'C'] (list answers of the enumeration questions)."""
    parts = re.split(r"\s*(?:,|;|\band\b|\n)\s*", answer)
    return [p for p in (x.strip() for x in parts) if normalize(p)]


def set_f1(pred: str, gold_items: list[str]) -> float:
    """F1 between the predicted list and the gold list, items compared after normalization."""
    p = {normalize(x) for x in split_list(pred)}
    g = {normalize(x) for x in gold_items}
    if not p or not g:
        return float(p == g)
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def contains(haystack: str, needle: str) -> bool:
    """Whether the normalized needle occurs in the normalized haystack on token boundaries."""
    n = normalize(needle)
    if not n:
        return False
    return f" {n} " in f" {normalize(haystack)} "


_WEAK = re.compile(
    r"^(yes|no|\d+(st|nd|rd|th)?|\d[\d,.\s]*|"
    r"(\d{1,2}\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"(\s+\d{1,2})?(\s+\d{3,4})?|\d{1,2}\s+\d{1,2}\s+\d{2,4})$")


def is_weak_span(span: str) -> bool:
    """Spans that can match by chance (years, numbers, dates, yes/no, very short strings)."""
    n = normalize(span)
    return len(n) <= 3 or bool(_WEAK.match(n))


def approx_tokens(s: str) -> int:
    """Rough token count (about 4 characters per token for English); use a real tokenizer when exactness matters."""
    return max(1, (len(s) + 3) // 4)


_NUM = re.compile(r"(?<![\w.])(\d{1,4})(?![\w])")


def standalone_numbers(s: str) -> set[int]:
    """Numbers of 1-4 digits that stand alone in the text (years, counts, page numbers): IDs must avoid them."""
    return {int(m) for m in _NUM.findall(s)}
