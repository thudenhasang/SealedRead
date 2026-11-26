"""The episode driver: runs the six-tag protocol for many episodes at once over any policy.

The driver owns everything that is not the model: the state machine, the forced alpha=answer at round R_max, the
checks on <locate>, the paste engine (<knowledge> in C order), the oracle and retriever locators of the S0 configs
and the RAG baseline, and the ID permutation probe (RQ4). A policy only writes text:
    gen_masked(eps, tag, force)  -> [(content, hit_cap)]   content of <tag>...</tag>, C masked
    gen_locate(eps, allowed)     -> [[ids]]                1..K IDs from `allowed`, C visible
Episodes are grouped by the tag they need next, so a batched policy (vLLM) gets one call per tag and round.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from srd.data.sample import Sample
from srd.protocol import (ProtoConfig, ProtocolError, Step, check_locate, force_alpha_answer, must_answer,
                          next_tag, normalize_alpha, normalize_beta, order_in_context, parse_alpha, parse_beta,
                          pasted, validate)


@dataclass
class Episode:
    sample: Sample
    steps: list[Step] = field(default_factory=list)
    done: bool = False
    error: str | None = None             # the violation that ended the episode (strict mode)
    seed: int = 0
    sample_idx: int = 0                  # which of the n samples of its question
    state: dict = field(default_factory=dict)       # backend state (KV cache, builder...)
    stats: dict = field(default_factory=lambda: {"format_fixes": 0, "cap_hits": 0, "dropped_ids": 0, "normalized": 0})

    def violations(self, cfg: ProtoConfig, token_count=None) -> list[str]:
        if self.error:
            return [self.error]
        return validate(self.steps, self.sample.ids(), self.sample.order(), cfg, token_count)


class Policy(Protocol):
    def gen_masked(self, eps: list[Episode], tag: str, force: list[bool]) -> list[tuple[str, bool]]: ...

    def gen_locate(self, eps: list[Episode], allowed: list[list[int]]) -> list[list[int]]: ...

    def release(self, ep: Episode) -> None: ...


@dataclass
class DriverConfig:
    proto: ProtoConfig = field(default_factory=ProtoConfig)
    locator: str = "model"               # model | retriever | oracle
    lenient: bool = False                # zero-shot runs: default a missing alpha/beta, keep going after a cap hit
    permute_ids: bool = False            # RQ4: relabel the pasted segments with fresh random IDs
    max_steps: int = 64

    def effective_proto(self) -> ProtoConfig:
        if self.locator == "oracle":     # G pasted at the first locate, then the model must answer
            p = self.proto
            return ProtoConfig(k=p.k, r_max=1, caps=p.caps, locate_max_tokens=p.locate_max_tokens)
        return self.proto


def _alias(sample: Sample, ids: list[int], used: set[int], rng: random.Random) -> dict[int, int]:
    """Fresh labels with the same number of digits (same token count), absent from C and from earlier aliases."""
    taken = sample.ids() | used
    out = {}
    for sid in ids:
        lo, hi = 10 ** (len(str(sid)) - 1), 10 ** len(str(sid)) - 1
        for _ in range(1000):
            x = rng.randint(max(lo, 1), hi)
            if x not in taken:
                break
        else:
            raise RuntimeError("no free alias")
        taken.add(x)
        used.add(x)
        out[sid] = x
    return out


def run_episodes(policy: Policy, samples: list[Sample], n: int, cfg: DriverConfig, retriever=None,
                 seed: int = 0) -> list[Episode]:
    """n episodes per sample; returns them in order (sample 0's n episodes, then sample 1's...)."""
    proto = cfg.effective_proto()
    eps = [Episode(s, seed=seed * 1_000_003 + i * 101 + j, sample_idx=j)
           for i, s in enumerate(samples) for j in range(n)]
    rngs = {id(e): random.Random(e.seed) for e in eps}
    used_alias: dict[int, set[int]] = {id(e): set() for e in eps}
    active = list(eps)
    while active:
        groups: dict[str, list[Episode]] = {}
        for ep in active:
            if len(ep.steps) >= cfg.max_steps:
                ep.error, ep.done = "too many steps", True
                continue
            try:
                tag = next_tag(ep.steps, proto)
            except ProtocolError as e:
                ep.error, ep.done = str(e), True
                continue
            if tag is None:
                ep.done = True
                continue
            groups.setdefault(tag, []).append(ep)
        for tag, group in groups.items():
            if tag == "locate":
                _locate(policy, group, cfg, proto, retriever, rngs, used_alias)
            elif tag == "knowledge":
                raise AssertionError("knowledge is written together with locate")
            else:
                force = [tag == "reason" and must_answer(ep.steps, proto) for ep in group]
                outs = policy.gen_masked(group, tag, force)
                for ep, f, (text, hit) in zip(group, force, outs):
                    if hit:
                        ep.stats["cap_hits"] += 1
                        if not cfg.lenient:
                            ep.error, ep.done = f"<{tag}> hit its cap", True
                            continue
                    text = _fix_format(ep, tag, text, f, cfg)
                    if text is None:
                        continue
                    ep.steps.append(Step(tag, text))
        active = [ep for ep in active if not ep.done]
    for ep in eps:
        policy.release(ep)
    return eps


def _fix_format(ep: Episode, tag: str, text: str, force: bool, cfg: DriverConfig) -> str | None:
    # D29: a decorated final alpha / beta ("**alpha=query**.") is rewritten canonically in every mode, so traces
    # stay clean; counted apart from the format fixes (which invent a missing value, lenient mode only)
    if tag in ("reason", "recheck"):
        text, changed = (normalize_alpha if tag == "reason" else normalize_beta)(text)
        if changed:
            ep.stats["normalized"] = ep.stats.get("normalized", 0) + 1
    if tag == "reason":
        if force and parse_alpha(text) != "answer":
            text = force_alpha_answer(text)
        if parse_alpha(text) is None:
            if not cfg.lenient:
                ep.error, ep.done = "reason without alpha", True
                return None
            ep.stats["format_fixes"] += 1
            text = (text.rstrip() + " alpha=query").strip()
    if tag == "recheck" and parse_beta(text) is None:
        if not cfg.lenient:
            ep.error, ep.done = "recheck without beta", True
            return None
        ep.stats["format_fixes"] += 1
        text = (text.rstrip() + " beta=valid_to_think").strip()
    return text


def _last_query(ep: Episode) -> str:
    for st in reversed(ep.steps):
        if st.tag == "query":
            return st.text
    return ep.sample.question


def _locate(policy, group, cfg: DriverConfig, proto: ProtoConfig, retriever, rngs, used_alias) -> None:
    allowed = []
    for ep in group:
        done = set(pasted(ep.steps))
        allowed.append([s.sid for s in ep.sample.segments if s.sid not in done])
    if cfg.locator == "model":
        live = [(ep, al) for ep, al in zip(group, allowed) if al]
        for ep, al in zip(group, allowed):
            if not al:
                ep.error, ep.done = "nothing left to locate", True
        outs = policy.gen_locate([e for e, _ in live], [a for _, a in live]) if live else []
        chosen = [(ep, ids, "model") for (ep, _), ids in zip(live, outs)]
    elif cfg.locator == "retriever":
        chosen = [(ep, retriever.top(ep.sample, _last_query(ep), proto.k, exclude=set(pasted(ep.steps))), "engine")
                  for ep in group]
    elif cfg.locator == "oracle":
        chosen = []
        for ep, al in zip(group, allowed):
            g = [s for s in ep.sample.gold if s in set(al)]
            if not g:
                ep.error, ep.done = "oracle locate without gold", True
                continue
            chosen.append((ep, g, "engine"))
    else:
        raise ValueError(f"unknown locator {cfg.locator!r}")
    for ep, ids, by in chosen:
        if by == "model":
            uniq = list(dict.fromkeys(ids))
            ep.stats["dropped_ids"] += len(ids) - len(uniq)
            ids = uniq
            err = check_locate(ids, ep.sample.ids(), set(pasted(ep.steps)), proto.k)
            if err:
                ep.error, ep.done = err, True
                continue
        if not ids:
            ep.error, ep.done = "empty locate", True
            continue
        order = order_in_context(ep.sample.order(), ids)
        alias = _alias(ep.sample, ids, used_alias[id(ep)], rngs[id(ep)]) if cfg.permute_ids else None
        ep.steps.append(Step("locate", ids=list(ids), by=by, alias=alias))
        ep.steps.append(Step("knowledge", ids=order, by="engine", alias=alias))
