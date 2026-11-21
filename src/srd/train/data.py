"""Traces -> training sequences, per branch. Every branch trains on the same trace set with its own mask (table 10).

    sealedread        tier 1 (two-stream engine): the masked view, compact frame (targets: masked tags) + the locate
                      view (targets: <locate> only). Matches srd.engine.vllm_policy mode two_stream token for token.
    sealedread_chain  tier 2: the single chain of the main design (full frame), sealedread mask
    nomask            the single chain where everything sees C (same text, same engine); vLLM mode nomask
    daft              the single chain, masked phase sees the original KV of the chosen segments, no pasted text
    rag               the masked view only (<locate> replaced by a retriever: no locate tokens to learn)
    sealong           plain full-attention CoT traces (question after C)
"""
from __future__ import annotations

from dataclasses import dataclass

from srd.data.sample import Sample, read_jsonl
from srd.layout import A, GEN, MaskOpts, Seq, SeqBuilder, build_seq
from srd.prompts import Prompts
from srd.protocol import Step, steps_from_dicts
from srd.tok import Tok

BRANCH_VIEWS: dict[str, list[tuple[str, dict]]] = {
    "sealedread": [("masked", {"frame": "compact"}), ("locate", {})],
    "sealedread_chain": [("chain", {"branch": "sealedread"})],
    "nomask": [("chain", {"branch": "nomask"})],
    "daft": [("chain", {"branch": "daft"})],
    "rag": [("masked", {"frame": "compact"})],
}
BRANCHES = tuple(BRANCH_VIEWS) + ("sealong",)


@dataclass
class TraceItem:
    sample: Sample
    steps: list[Step] | None = None      # protocol traces
    text: str | None = None              # sealong: the full-attention response
    meta: dict | None = None


def plain_seq(s: Sample, response: str, tok: Tok, prompts: Prompts) -> Seq:
    """Question-after-C prompt (no targets) + the response (targets)."""
    b = SeqBuilder(tok, MaskOpts("nomask"), "chain")
    b.add(prompts.plain(s), A)
    b.add_pieces([response, prompts.eos], GEN, target=True)
    return b.seq()


def sequences(item: TraceItem, tok: Tok, prompts: Prompts, branch: str) -> list[Seq]:
    if branch == "sealong":
        return [plain_seq(item.sample, item.text or "", tok, prompts)]
    if branch not in BRANCH_VIEWS:
        raise ValueError(f"unknown branch {branch!r}")
    out = []
    for view, kw in BRANCH_VIEWS[branch]:
        seq = build_seq(item.sample, item.steps or [], tok, prompts, view=view, **kw)
        if seq.target.any():
            out.append(seq)
    return out


def load_traces(path: str, samples: dict[str, Sample], limit: int | None = None) -> list[TraceItem]:
    """Trace files: one JSON per line, {"qid", "steps"} (protocol) or {"qid", "text"} (sealong)."""
    out = []
    for d in read_jsonl(path):
        s = samples.get(d["qid"])
        if s is None:
            raise KeyError(f"trace for unknown sample {d['qid']}")
        steps = steps_from_dicts(d["steps"]) if "steps" in d else None
        out.append(TraceItem(s, steps, d.get("text"), {k: v for k, v in d.items() if k not in ("steps", "text")}))
        if limit and len(out) >= limit:
            break
    return out


def sample_index(paths: list[str]) -> dict[str, Sample]:
    from srd.data.sample import load_samples

    idx: dict[str, Sample] = {}
    for p in paths:
        for s in load_samples(p):
            idx[s.qid] = s
    return idx
