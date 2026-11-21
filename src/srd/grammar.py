"""The <locate> grammar: 1..K distinct IDs that exist in C and were not pasted yet, then </locate>.

Two forms of the same rule:
    LocateTrie    exact, token level (canonical pieces), for the reference engine: distinctness included
    locate_regex  for vLLM guided decoding; a regex cannot forbid repeats, so the engine drops repeated IDs and
                  counts them (docs/decisions.md D5)
"""
from __future__ import annotations

from srd.protocol import label
from srd.tok import Tok


class _Node:
    __slots__ = ("children", "sids", "terminal")

    def __init__(self):
        self.children: dict[int, _Node] = {}
        self.sids: set[int] = set()
        self.terminal: int | None = None


class LocateTrie:
    """Token-level constraint. Feed the tokens after '<locate>'; `allowed()` lists the legal next tokens."""

    def __init__(self, tok: Tok, allowed_ids: list[int] | set[int], k: int):
        if not allowed_ids:
            raise ValueError("no ID left to locate")
        self.k = k
        self.root = _Node()
        for sid in allowed_ids:
            toks = tok.encode(label(sid))
            node = self.root
            node.sids.add(sid)
            for t in toks:
                node = node.children.setdefault(t, _Node())
                node.sids.add(sid)
                if node.terminal is not None:
                    raise ValueError("label tokenizations are not prefix-free")
            if node.children:
                raise ValueError("label tokenizations are not prefix-free")
            node.terminal = sid
        self.sep = tok.encode(" ")
        self.end = tok.encode("</locate>")
        if not self.sep or not self.end or self.sep[0] == self.end[0]:
            raise ValueError("separator and end tag must start with different tokens")
        self.chosen: list[int] = []
        self.phase = "item"       # item | between | sep | end | done
        self.node = self.root
        self.i = 0                # position inside sep / end

    def _open(self, node: _Node) -> list[int]:
        free = [t for t, ch in node.children.items() if ch.sids - set(self.chosen)]
        return free

    def allowed(self) -> list[int]:
        if self.phase == "item":
            return self._open(self.node)
        if self.phase == "between":
            out = [self.end[0]]
            if len(self.chosen) < self.k and self._open(self.root):
                out.append(self.sep[0])
            return out
        if self.phase == "sep":
            return [self.sep[self.i]]
        if self.phase == "end":
            return [self.end[self.i]]
        return []

    def advance(self, t: int) -> None:
        if t not in self.allowed():
            raise ValueError(f"token {t} not allowed in phase {self.phase}")
        if self.phase == "item":
            self.node = self.node.children[t]
            if self.node.terminal is not None:
                self.chosen.append(self.node.terminal)
                self.phase = "between"
        elif self.phase == "between":
            if t == self.end[0]:
                self.phase, self.i = "end", 0
            else:
                self.phase, self.i = "sep", 0
            self._step_forced()
        elif self.phase in ("sep", "end"):
            self._step_forced()

    def _step_forced(self) -> None:
        seq = self.sep if self.phase == "sep" else self.end
        self.i += 1
        if self.i >= len(seq):
            if self.phase == "sep":
                self.phase, self.node = "item", self.root
            else:
                self.phase = "done"

    @property
    def done(self) -> bool:
        return self.phase == "done"


def locate_regex(allowed_ids: list[int] | set[int], k: int, with_end: bool = True) -> str:
    """'[[S_a]]( [[S_b]]){0,k-1}</locate>' over the allowed IDs (vLLM structured output)."""
    alt = "|".join(str(i) for i in sorted(allowed_ids))
    item = rf"\[\[S_(?:{alt})\]\]"
    rx = rf"{item}(?: {item}){{0,{k - 1}}}"
    return rx + ("</locate>" if with_end else "")
