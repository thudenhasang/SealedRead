"""Real-document benchmarks for RQ2 and out-of-distribution sets, converted to Sample.

    QASPER      papers -> paragraphs (section names kept); answer-F1 (max over annotators); G = evidence paragraphs;
                padded with papers of the same split to 32K / 64K / 128K
    NarrativeQA full stories (paragraphs = blank-line blocks); token-F1; kept only if the input fits 128K tokens
    QuALITY     articles + 4-option questions (answer: the letter or the option text); accuracy; padded
When several documents share a context, questions that say "the authors" / "they" are rewritten to name the
document ("In the paper 'X', ..."), so they are not ambiguous (section 6.2).

The converters take records in the layout of the Hugging Face copies named in configs/benchmarks.yaml; check the
field names on the first run (scripts/fetch_benchmarks.py prints one converted sample per benchmark).
"""
from __future__ import annotations

import random
import re
from typing import Callable, Iterable

from srd.data.chunk import Chunk, merge_short, split_sentences
from srd.data.ids import assign_ids, forbidden_ids
from srd.data.sample import Sample, Segment
from srd.text import approx_tokens, normalize

TokenCounter = Callable[[str], int]


def doc_chunks(doc: str, title: str, paragraphs: list[tuple[str, str]], count: TokenCounter, max_tokens: int = 256,
               min_tokens: int = 32, gold_texts: set[str] | None = None) -> list[Chunk]:
    out = []
    gold_norm = {normalize(g) for g in (gold_texts or set()) if g}
    for section, text in paragraphs:
        text = " ".join(text.split())
        if not text:
            continue
        sents = split_sentences(text)
        cur, size = [], 0
        for s in sents:
            n = count(s) + 1
            if cur and size + n > max_tokens:
                out.append(Chunk(title, " ".join(cur), doc, section or "Text"))
                cur, size = [], 0
            cur.append(s)
            size += n
        if cur:
            out.append(Chunk(title, " ".join(cur), doc, section or "Text"))
    out = merge_short(out, count, min_tokens, max_tokens)
    for c in out:
        nc = normalize(c.text)
        c.gold = any(g and (g in nc or nc in g) for g in gold_norm)
    return out


def assemble(qid: str, dataset: str, question: str, answers: list[str], answer_kind: str, main: list[Chunk],
             pads: Iterable[list[Chunk]], target: int, count: TokenCounter, rng: random.Random,
             cluster: str) -> Sample | None:
    """The main document's chunks (in order) plus whole padding documents, in random document order."""
    used = sum(count(c.text) + 16 for c in main)
    if used > target:
        return None
    docs = [main]
    for d in pads:
        cost = sum(count(c.text) + 16 for c in d)
        if used + cost > target:
            continue
        docs.append(d)
        used += cost
        if used >= target * 0.97:
            break
    rng.shuffle(docs)
    chunks = [c for d in docs for c in d]
    ids = assign_ids(len(chunks), rng, forbidden_ids([c.text for c in chunks] + [question]))
    segs = [Segment(i, c.title, c.section, c.text, c.doc, c.gold) for i, c in zip(ids, chunks)]
    return Sample(qid=qid, dataset=dataset, qtype="benchmark", question=question, answers=answers,
                  answer_kind=answer_kind, segments=segs, cluster=cluster, length=target, meta={"tokens": used})


_PRONOUN = re.compile(r"\b(the authors|they|their|this paper|the paper|this work)\b", re.I)


def disambiguate(question: str, title: str, kind: str = "paper") -> str:
    if _PRONOUN.search(question):
        return f"In the {kind} '{title}', {question[0].lower() + question[1:]}"
    return question


# ---------------------------------------------------------------------- QASPER
def qasper_docs(records: Iterable[dict]) -> list[dict]:
    out = []
    for r in records:
        ft = r["full_text"]
        paras = [("Abstract", r.get("abstract", ""))]
        for sec, ps in zip(ft["section_name"], ft["paragraphs"]):
            paras += [(sec or "Text", p) for p in ps]
        qas = []
        for q, ans in zip(r["qas"]["question"], r["qas"]["answers"]):
            refs, ev = [], set()
            for a in ans["answer"]:
                if a.get("unanswerable"):
                    refs.append("Unanswerable")
                elif a.get("yes_no") is not None:
                    refs.append("Yes" if a["yes_no"] else "No")
                elif a.get("extractive_spans"):
                    refs.append(", ".join(a["extractive_spans"]))
                elif a.get("free_form_answer"):
                    refs.append(a["free_form_answer"])
                ev.update(a.get("evidence") or [])
            if refs:
                closed = all(normalize(x) in ("yes", "no") for x in refs)
                qas.append({"question": q, "answers": refs, "evidence": ev, "kind": "closed" if closed else "open"})
        out.append({"doc": r["id"], "title": r["title"], "paragraphs": paras, "qas": qas})
    return out


def build_docs_samples(docs: list[dict], dataset: str, targets: list[int], count: TokenCounter = approx_tokens,
                       seed: int = 0, limit: int | None = None, kind: str = "paper") -> list[Sample]:
    rng = random.Random(seed)
    chunked = {d["doc"]: doc_chunks(d["doc"], d["title"], d["paragraphs"], count) for d in docs}
    out = []
    for d in docs:
        for qi, qa in enumerate(d["qas"]):
            for target in targets:
                main = doc_chunks(d["doc"], d["title"], d["paragraphs"], count, gold_texts=qa.get("evidence"))
                others = [v for k, v in chunked.items() if k != d["doc"]]
                rng.shuffle(others)
                q = disambiguate(qa["question"], d["title"], kind) if target else qa["question"]
                s = assemble(f"{dataset}-{d['doc']}-{qi}-{target}", dataset, q, qa["answers"], qa["kind"], main,
                             others, target, count, rng, cluster=f"{dataset}-{d['doc']}")
                if s is not None:
                    out.append(s)
            if limit and len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------- NarrativeQA
def narrativeqa_docs(records: Iterable[dict], max_words: int = 90000) -> list[dict]:
    docs: dict[str, dict] = {}
    for r in records:
        d = r["document"]
        if d.get("word_count", 0) and d["word_count"] > max_words:
            continue
        if d["id"] not in docs:
            blocks = [b for b in re.split(r"\n\s*\n", d["text"]) if b.strip()]
            title = (d.get("summary") or {}).get("title") or d["id"]
            docs[d["id"]] = {"doc": d["id"], "title": title, "paragraphs": [("Story", b) for b in blocks], "qas": []}
        docs[d["id"]]["qas"].append({"question": r["question"]["text"], "answers": [a["text"] for a in r["answers"]],
                                     "evidence": set(), "kind": "open"})
    return list(docs.values())


# ---------------------------------------------------------------------- QuALITY
def quality_docs(records: Iterable[dict]) -> list[dict]:
    docs: dict[str, dict] = {}
    for r in records:
        aid = str(r["article_id"])
        if aid not in docs:
            text = re.sub(r"<[^>]+>", " ", r["article"])
            blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
            docs[aid] = {"doc": aid, "title": r.get("title", aid), "paragraphs": [("Article", b) for b in blocks],
                         "qas": []}
        qs = r["questions"] if "questions" in r else [r]
        for q in qs:
            opts = q["options"]
            letters = "ABCD"
            gold = int(q["gold_label"]) - 1
            question = q["question"] + "\n" + "\n".join(f"({letters[i]}) {o}" for i, o in enumerate(opts)) + \
                "\nAnswer with the letter of the right option."
            docs[aid]["qas"].append({"question": question, "answers": [letters[gold], f"({letters[gold]})"],
                                     "evidence": set(), "kind": "closed"})
    return list(docs.values())
