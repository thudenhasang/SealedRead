"""Reference engine: our Qwen3 + the layout's own masks, one episode at a time.

It runs every branch of table 10 and every inference ablation exactly as the trainer sees them (it builds its
sequence with the same SeqBuilder, and tests/test_engine.py checks that a forced trace gives the very layout of
build_seq). It is slow, and it is the yardstick: S0 configs that need no C prefill ((b), (c), (f): `view="masked"`
with an oracle locator), the invariance probes, and the log-prob parity check of the fast engine (check 6).

Canonical tokenization: after a tag closes, the generated tokens are rolled back and the step is re-encoded from its
text pieces, so the cache always holds exactly the tokens the trainer would build for the same trace.
"""
from __future__ import annotations

import numpy as np
import torch

from srd.engine.driver import Episode
from srd.grammar import LocateTrie
from srd.layout import C, CPY, GEN, LOC, QP, MaskOpts, SeqBuilder, add_prompt, add_step, copy_pieces, visible, \
    tag_pieces
from srd.model.attention import Attn
from srd.prompts import Prompts
from srd.protocol import ProtoConfig, Step, force_alpha_answer
from srd.tok import Tok


class _Mirror:
    """numpy copies of the builder's per-token metadata, extended incrementally."""

    def __init__(self):
        self.n = 0
        self.version = -1
        self.a = {k: np.zeros(0, np.int64) for k in ("kind", "rnd", "open_at", "stp", "qref")}

    def get(self, b: SeqBuilder, hi: int) -> dict[str, np.ndarray]:
        if b.version != self.version or self.n > len(b):
            self.n, self.version = 0, b.version
        if self.n < len(b):
            for k in self.a:
                tail = np.asarray(getattr(b, k)[self.n:], np.int64)
                self.a[k] = np.concatenate([self.a[k][:self.n], tail])
            self.n = len(b)
        return {k: v[:hi] for k, v in self.a.items()}

    def truncate(self, n: int) -> None:
        self.n = min(self.n, n)


class RefPolicy:
    def __init__(self, model, tok: Tok, prompts: Prompts | None = None, branch: str = "sealedread",
                 view: str = "chain", frame: str = "full", repeat_q: bool = True, opts: MaskOpts | None = None,
                 proto: ProtoConfig | None = None, temperature: float = 0.0, top_p: float = 1.0):
        if view not in ("chain", "masked"):
            raise ValueError("the reference engine runs the chain, or the masked view with an engine locator")
        self.model = model
        self.tok = tok
        self.prompts = prompts or Prompts()
        self.opts = opts or MaskOpts(branch=branch)
        self.view = view
        self.frame = frame
        self.repeat_q = repeat_q
        self.proto = proto or ProtoConfig()
        self.temperature = temperature
        self.top_p = top_p
        self.eos = tok.eos_ids()

    # ------------------------------------------------------------------ state
    def _state(self, ep: Episode) -> dict:
        st = ep.state
        if "b" in st:
            return st
        b = SeqBuilder(self.tok, self.opts, self.view)
        add_prompt(b, ep.sample, self.prompts, with_c=self.view == "chain", frame=self.frame, repeat_q=self.repeat_q)
        g = torch.Generator()
        g.manual_seed(ep.seed)
        st.update(b=b, cache=self.model.new_cache(), mirror=_Mirror(), fed=0, h=None, synced=0, pending=None, gen=g)
        n_pre = next(i for i, k in enumerate(b.kind) if k >= QP)
        self._feed(st, upto=n_pre, causal=True)
        self._feed(st)
        return st

    @torch.no_grad()
    def _feed(self, st: dict, upto: int | None = None, causal: bool = False) -> None:
        b: SeqBuilder = st["b"]
        lo, hi = st["fed"], len(b) if upto is None else upto
        if hi <= lo:
            return
        dev = self.model.device
        ids = torch.as_tensor(b.ids[lo:hi], device=dev)[None]
        pos = torch.as_tensor(b.pos[lo:hi], device=dev)[None]
        if causal and lo == 0:
            spec = Attn("causal")
        else:
            a = st["mirror"].get(b, hi)
            rows = visible(np.arange(lo, hi), a["kind"][lo:hi], a["rnd"][lo:hi], a["qref"][lo:hi], np.arange(hi),
                           a["kind"], a["rnd"], a["open_at"], a["stp"], self.opts)
            keep = np.nonzero(rows.any(axis=0))[0]
            spec = Attn("dense", mask=torch.as_tensor(rows[:, keep], device=dev),
                        kv_index=torch.as_tensor(keep, device=dev))
        h = self.model(ids, pos, spec, st["cache"])
        st["h"] = h[0, -1]
        st["fed"] = hi

    def _truncate(self, st: dict, n: int) -> None:
        st["b"].truncate(n)
        st["cache"].truncate(n)
        st["mirror"].truncate(n)
        st["fed"] = n

    def _sync(self, ep: Episode) -> dict:
        """Encode the steps the policy did not write itself (pasted text, copies, forced prefixes)."""
        st = self._state(ep)
        b: SeqBuilder = st["b"]
        last = st.get("last")
        if last is not None:
            idx, start, content = last
            st["last"] = None
            if idx < len(ep.steps) and ep.steps[idx].text != content:   # the driver fixed the format
                self._truncate(st, start)
                st["synced"] = idx
            elif idx >= len(ep.steps):                                     # the driver dropped the step
                self._truncate(st, start)
                st["synced"] = len(ep.steps)
        for i in range(st["synced"], len(ep.steps)):
            step = ep.steps[i]
            if step.tag == "locate" and step.by == "model" and st["pending"] is not None:
                lo, hi = st["pending"]
                b.step = i
                cs, _ = b.add_pieces(copy_pieces(step), CPY, pos=b.pos[lo:hi])
                if self.opts.branch == "daft" and self.view == "chain":
                    b.open(step.ids, at=cs)
                st["pending"] = None
            else:
                add_step(b, ep.sample, step, i, self.prompts)
        st["synced"] = len(ep.steps)
        self._feed(st)
        return st

    def release(self, ep: Episode) -> None:
        ep.state.clear()

    # ------------------------------------------------------------------ sampling
    def _sample(self, st: dict, allowed: list[int] | None = None) -> int:
        logits = self.model.logits(st["h"][None])[0].float().cpu()
        if allowed is not None:
            m = torch.full_like(logits, float("-inf"))
            m[allowed] = 0
            logits = logits + m
        if self.temperature <= 0:
            return int(logits.argmax())
        probs = torch.softmax(logits / self.temperature, -1)
        if self.top_p < 1:
            sp, si = probs.sort(descending=True)
            cut = sp.cumsum(0) - sp > self.top_p
            sp[cut] = 0
            probs = torch.zeros_like(probs).scatter(0, si, sp)
        return int(torch.multinomial(probs / probs.sum(), 1, generator=st["gen"]))

    def _append(self, st: dict, t: int, kind: int) -> None:
        st["b"].add_ids([t], kind, target=True)
        self._feed(st)

    # ------------------------------------------------------------------ policy API
    def gen_masked(self, eps: list[Episode], tag: str, force: list[bool]) -> list[tuple[str, bool]]:
        out = []
        close = f"</{tag}>"
        for ep, f in zip(eps, force):
            st = self._sync(ep)
            b: SeqBuilder = st["b"]
            b.step = len(ep.steps)
            if tag == "query":
                b.cur_qref = len(ep.steps)
            open_at = len(b)
            b.add(f"<{tag}>", GEN, target=True)
            self._feed(st)
            start = len(b)
            gen: list[int] = []
            hit = True
            cap = self.proto.caps.get(tag, 512) + 16
            for _ in range(cap):
                t = self._sample(st)
                gen.append(t)
                self._append(st, t, GEN)
                if close in self.tok.decode(gen[-24:]):
                    hit = False
                    break
                if t in self.eos:
                    break
            text = self.tok.decode(gen)
            content = text.split(close)[0] if close in text else text
            for e in ("<|im_end|>", "<|endoftext|>"):
                content = content.replace(e, "")
            content = content.strip()
            if f and tag == "reason":
                content = force_alpha_answer(content)
            self._truncate(st, start)
            b.add_pieces(tag_pieces(tag, content, self.prompts)[1:], GEN, target=True)
            self._feed(st)
            st["synced"] = len(ep.steps) + 1
            st["last"] = (len(ep.steps), open_at, content)
            out.append((content, hit))
        return out

    def gen_locate(self, eps: list[Episode], allowed: list[list[int]]) -> list[list[int]]:
        if self.view != "chain":
            raise ValueError("the masked view has no <locate> tokens: use an oracle or retriever locator")
        out = []
        for ep, al in zip(eps, allowed):
            st = self._sync(ep)
            b: SeqBuilder = st["b"]
            b.step = len(ep.steps)
            b.round += 1
            start, _ = b.add("<locate>", LOC, target=True)
            self._feed(st)
            trie = LocateTrie(self.tok, al, self.proto.k)
            while not trie.done:
                t = self._sample(st, trie.allowed())
                trie.advance(t)
                self._append(st, t, LOC)
            st["pending"] = (start, len(b))
            st["synced"] = len(ep.steps)          # the copy is written by _sync once the step exists
            out.append(list(trie.chosen))
        return out

    # ------------------------------------------------------------------ forcing (probes, tests)
    def force(self, ep: Episode, steps: list[Step]) -> SeqBuilder:
        """Encode a given trace (teacher forcing) and return the builder; its layout equals build_seq's."""
        ep.steps = list(steps)
        st = self._sync(ep)
        return st["b"]


def c_token_count(b: SeqBuilder) -> int:
    return sum(1 for k in b.kind if k == C)
