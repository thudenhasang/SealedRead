"""Full-attention baselines: question after C (S0 config (a), SEALONG), copy-then-solve, one-pass RAG.

They use the plain chat format: <reason> ... </reason> <answer> ... </answer>, no tags that touch C.
"""
from __future__ import annotations

import re

from srd.data.sample import Sample
from srd.prompts import Prompts
from srd.tok import Tok

_ANS = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.S)


def parse_answer(text: str) -> str:
    m = _ANS.search(text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def plain_prompt(tok: Tok, prompts: Prompts, s: Sample, style: str = "plain") -> list[int]:
    return tok.encode(prompts.plain(s, style))


def run_plain(engine, tok: Tok, prompts: Prompts, samples: list[Sample], style: str = "plain",
              max_tokens: int = 1024, temperature: float = 0.0, n: int = 1, seed: int = 0) -> list[dict]:
    """One generation per (sample, j); returns records {qid, j, text, answer}."""
    reqs = [(s, j) for s in samples for j in range(n)]
    outs = engine.generate([plain_prompt(tok, prompts, s, style) for s, _ in reqs], max_tokens=max_tokens,
                           temperature=temperature, stop=["</answer>"],
                           seeds=[seed * 1_000_003 + i for i in range(len(reqs))])
    return [{"qid": s.qid, "j": j, "text": o.text + "</answer>", "answer": parse_answer(o.text + "</answer>"),
             "truncated": o.finish_reason == "length"} for (s, j), o in zip(reqs, outs)]


def run_rag_one_pass(engine, tok: Tok, prompts: Prompts, samples: list[Sample], retriever, k: int,
                     max_tokens: int = 1024) -> list[dict]:
    reqs = []
    for s in samples:
        ids = retriever.top(s, s.question, k)
        order = s.order()
        segs = [s.by_sid(i) for i in sorted(ids, key=lambda x: order[x])]
        reqs.append((s, ids, tok.encode(prompts.rag_one_pass(s, segs))))
    outs = engine.generate([p for _, _, p in reqs], max_tokens=max_tokens, temperature=0.0, stop=["</answer>"])
    return [{"qid": s.qid, "j": 0, "ids": ids, "text": o.text, "answer": parse_answer(o.text + "</answer>"),
             "truncated": o.finish_reason == "length"} for (s, ids, _), o in zip(reqs, outs)]
