"""The six-tag protocol at the text level: steps, the state machine, validity rules and the paste engine.

A solution is a flat list of steps. The model writes <reason>, <query>, <recheck>, <answer> (the masked phase) and
<locate> (the only tag that sees C, and can only emit 1..K IDs); the engine writes <knowledge> (the pasted segments,
in their order in C). Section 3.2 of the proposal, with the two fixes recorded in docs/decisions.md:
    D2  a <locate> copy keeps the original's position IDs and length; its content is the IDs sorted by value
    D3  at round R_max, beta=reject_and_requery leads to <reason> (forced to alpha=answer), not to <query>
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

TAGS = ("reason", "query", "locate", "knowledge", "recheck", "answer")
MASKED_TAGS = ("reason", "query", "recheck", "answer")      # written by the model without seeing C
ALPHAS = ("query", "answer")
BETAS = ("valid_to_think", "explore_query", "reject_and_requery")
ID_MIN, ID_MAX = 1, 9999

_ALPHA_RE = re.compile(r"alpha\s*=\s*(query|answer)\s*$")
_BETA_RE = re.compile(r"beta\s*=\s*(valid_to_think|explore_query|reject_and_requery)\s*$")
_ID_RE = re.compile(r"\[\[S_(\d{1,4})\]\]")


class ProtocolError(ValueError):
    pass


def label(sid: int) -> str:
    return f"[[S_{sid}]]"


def close_label(sid: int) -> str:
    return f"[[/S_{sid}]]"


def locate_pieces(ids: list[int]) -> list[str]:
    """Content of a <locate> block as pieces: '[[S_871]]', ' ', '[[S_2305]]' (canonical tokenization)."""
    out: list[str] = []
    for i, sid in enumerate(ids):
        if i:
            out.append(" ")
        out.append(label(sid))
    return out


def parse_ids(text: str) -> list[int]:
    return [int(x) for x in _ID_RE.findall(text)]


def parse_alpha(text: str) -> str | None:
    m = _ALPHA_RE.search(text.strip())
    return m.group(1) if m else None


def parse_beta(text: str) -> str | None:
    m = _BETA_RE.search(text.strip())
    return m.group(1) if m else None


def force_alpha_answer(text: str) -> str:
    """At round R_max the grammar only lets <reason> end with alpha=answer: rewrite or append the final alpha."""
    t = text.rstrip()
    if _ALPHA_RE.search(t):
        return _ALPHA_RE.sub("alpha=answer", t)
    return (t + " alpha=answer").strip()


@dataclass
class Step:
    tag: str
    text: str = ""                       # content between the tags (masked tags); unused for locate / knowledge
    ids: list[int] = field(default_factory=list)   # locate: IDs in generated order; knowledge: pasted, C order
    by: str = "model"                    # "model", or "engine" (knowledge; oracle / retriever locate)
    alias: dict[int, int] | None = None  # ID permutation probe: labels shown in the copy and the pasted block

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["alias"] is not None:
            d["alias"] = {str(k): v for k, v in d["alias"].items()}
        return d

    @staticmethod
    def from_dict(d: dict) -> "Step":
        alias = d.get("alias")
        return Step(tag=d["tag"], text=d.get("text", ""), ids=list(d.get("ids", [])), by=d.get("by", "model"),
                    alias={int(k): int(v) for k, v in alias.items()} if alias else None)


def steps_to_dicts(steps: list[Step]) -> list[dict]:
    return [s.to_dict() for s in steps]


def steps_from_dicts(ds: list[dict]) -> list[Step]:
    return [Step.from_dict(d) for d in ds]


@dataclass
class ProtoConfig:
    k: int = 3                           # IDs per <locate>
    r_max: int = 8                       # rounds; after round r_max the next <reason> must answer
    caps: dict = field(default_factory=lambda: {"reason": 512, "query": 64, "recheck": 256, "answer": 64})
    locate_max_tokens: int = 48          # safety cap only (the grammar fixes the format)

    @staticmethod
    def from_cfg(d: dict | None) -> "ProtoConfig":
        d = dict(d or {})
        caps = {**ProtoConfig().caps, **(d.pop("caps", None) or {})}
        return ProtoConfig(caps=caps, **{k: v for k, v in d.items() if k in ("k", "r_max", "locate_max_tokens")})


# ---------------------------------------------------------------------- state machine
def rounds(steps: list[Step]) -> int:
    return sum(1 for s in steps if s.tag == "locate")


def pasted(steps: list[Step]) -> list[int]:
    out: list[int] = []
    for s in steps:
        if s.tag == "knowledge":
            out.extend(s.ids)
    return out


def located(steps: list[Step]) -> list[int]:
    out: list[int] = []
    for s in steps:
        if s.tag == "locate":
            out.extend(s.ids)
    return out


def must_answer(steps: list[Step], cfg: ProtoConfig) -> bool:
    """The next <reason> may only end with alpha=answer."""
    return rounds(steps) >= cfg.r_max


def next_tag(steps: list[Step], cfg: ProtoConfig) -> str | None:
    """The tag that must come next (None: the solution is complete). Raises ProtocolError on a broken history."""
    if not steps:
        return "reason"
    last = steps[-1]
    t = last.tag
    if t == "reason":
        a = parse_alpha(last.text)
        if a is None:
            raise ProtocolError("reason without alpha")
        if a == "answer":
            return "answer"
        if rounds(steps) >= cfg.r_max:
            raise ProtocolError("alpha=query after r_max")
        return "query"
    if t == "query":
        return "locate"
    if t == "locate":
        return "knowledge"
    if t == "knowledge":
        return "recheck"
    if t == "recheck":
        b = parse_beta(last.text)
        if b is None:
            raise ProtocolError("recheck without beta")
        if b == "reject_and_requery" and rounds(steps) < cfg.r_max:
            return "query"
        return "reason"
    if t == "answer":
        return None
    raise ProtocolError(f"unknown tag {t!r}")


def order_in_context(order: dict[int, int], ids: list[int]) -> list[int]:
    """The paste order: the segments as they appear in C."""
    return sorted(ids, key=lambda s: order[s])


def check_locate(ids: list[int], existing: set[int], already: set[int], k: int) -> str | None:
    if not 1 <= len(ids) <= k:
        return f"locate with {len(ids)} IDs"
    if len(set(ids)) != len(ids):
        return "locate repeats an ID"
    for s in ids:
        if s not in existing:
            return f"locate ID {s} does not exist"
        if s in already:
            return f"locate ID {s} was already pasted"
    return None


def validate(steps: list[Step], existing: set[int], order: dict[int, int], cfg: ProtoConfig,
             token_count=None) -> list[str]:
    """Every protocol violation of a finished solution (empty list: valid). Section 5.1, item 1.

    `token_count(text) -> int` (the student's tokenizer) adds the per-tag caps and the <locate> safety cap.
    """
    errs: list[str] = []
    hist: list[Step] = []
    already: set[int] = set()
    for i, st in enumerate(steps):
        try:
            want = next_tag(hist, cfg)
        except ProtocolError as e:
            errs.append(f"step {i}: {e}")
            break
        if want is None:
            errs.append(f"step {i}: content after <answer>")
            break
        if st.tag != want:
            errs.append(f"step {i}: <{st.tag}> where <{want}> was due")
            break
        if st.tag == "reason" and must_answer(hist, cfg) and parse_alpha(st.text) != "answer":
            errs.append(f"step {i}: reason after round r_max must answer")
        if st.tag == "locate":
            e = check_locate(st.ids, existing, already, cfg.k)
            if e:
                errs.append(f"step {i}: {e}")
            if token_count is not None and st.by == "model":
                n = token_count("".join(locate_pieces(st.ids)))
                if n > cfg.locate_max_tokens:
                    errs.append(f"step {i}: locate over {cfg.locate_max_tokens} tokens")
        if st.tag == "knowledge":
            prev = hist[-1] if hist else None
            new = [s for s in dict.fromkeys(prev.ids) if s in order and s not in already] if prev else []
            want_ids = order_in_context(order, new)
            if st.by != "engine" or list(st.ids) != want_ids:
                errs.append(f"step {i}: knowledge not written by the engine as pasted")
            already.update(st.ids)
        if st.tag in MASKED_TAGS and token_count is not None:
            cap = cfg.caps.get(st.tag)
            if cap and token_count(st.text) > cap:
                errs.append(f"step {i}: <{st.tag}> over its cap of {cap} tokens")
        hist.append(st)
    else:
        try:
            if next_tag(hist, cfg) is not None:
                errs.append("solution does not end with <answer>")
        except ProtocolError as e:
            errs.append(str(e))
    if rounds(steps) > cfg.r_max:
        errs.append(f"{rounds(steps)} rounds > r_max")
    if sum(1 for s in steps if s.tag == "answer") != 1:
        errs.append("not exactly one <answer>")
    return errs


def answer_of(steps: list[Step]) -> str:
    for s in reversed(steps):
        if s.tag == "answer":
            return s.text.strip()
    return ""


def masked_text(steps: list[Step]) -> str:
    """Everything the model wrote in the masked phase (reason, query, recheck, answer)."""
    return "\n".join(s.text for s in steps if s.tag in MASKED_TAGS)
