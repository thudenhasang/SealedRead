"""Fast policy on stock vLLM (tier 1 of the implementation plan, docs/decisions.md D1).

Modes:
    two_stream  SealedRead. Masked tags are written in a stream that holds no C at all ([A][Q][Q'][T'] + history,
                compact positions: appendix A); <locate> is written in a stream that holds C ([A][Q][C][Q'][T'] +
                history). Information-equivalent to the single chain (section 3.7); positions are the compact frame,
                so Delta_mask measured with it is the joint effect of masking and the position shift.
    nomask      the no-mask control. One stream with C. Dropping the hidden <locate> originals and keeping their
                copies in place gives exactly the KV of the single chain (layout.Seq.prompt_ids).
    masked      the masked stream only, for engine locators (RAG with a retriever; oracle configs).
Every prompt is built by srd.layout (canonical pieces), the same sequences the trainer uses.
"""
from __future__ import annotations

from srd.engine.driver import Episode
from srd.engine.vllm_engine import VLLMEngine
from srd.grammar import locate_regex
from srd.layout import build_seq
from srd.prompts import Prompts
from srd.protocol import ProtoConfig, force_alpha_answer, parse_ids
from srd.tok import Tok


class VLLMPolicy:
    def __init__(self, engine: VLLMEngine, tok: Tok, prompts: Prompts | None = None, mode: str = "two_stream",
                 proto: ProtoConfig | None = None, temperature: float = 0.0, top_p: float = 1.0,
                 repeat_q: bool = True, locate_query_only: bool = False):
        if mode not in ("two_stream", "nomask", "masked"):
            raise ValueError(mode)
        self.engine = engine
        self.tok = tok
        self.prompts = prompts or Prompts()
        self.mode = mode
        self.proto = proto or ProtoConfig()
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_q = repeat_q
        self.locate_query_only = locate_query_only

    def _masked_prompt(self, ep: Episode, tag: str) -> list[int]:
        if self.mode == "nomask":
            seq = build_seq(ep.sample, ep.steps, self.tok, self.prompts, view="locate", repeat_q=self.repeat_q)
        else:
            seq = build_seq(ep.sample, ep.steps, self.tok, self.prompts, view="masked", frame="compact",
                            repeat_q=self.repeat_q)
        return seq.prompt_ids() + self.tok.encode(f"<{tag}>")

    def _locate_prompt(self, ep: Episode) -> list[int]:
        steps = ep.steps
        if self.locate_query_only:                   # ablation: <locate> reads only its own <query>
            steps = [s for s in steps if s.tag == "query"][-1:]
        seq = build_seq(ep.sample, steps, self.tok, self.prompts, view="locate", repeat_q=self.repeat_q)
        return seq.prompt_ids() + self.tok.encode("<locate>")

    def gen_masked(self, eps: list[Episode], tag: str, force: list[bool]) -> list[tuple[str, bool]]:
        prompts = [self._masked_prompt(ep, tag) for ep in eps]
        cap = self.proto.caps.get(tag, 512)
        outs = self.engine.generate(prompts, max_tokens=cap + 8, temperature=self.temperature, top_p=self.top_p,
                                    stop=[f"</{tag}>"], seeds=[ep.seed * 7919 + len(ep.steps) for ep in eps])
        res = []
        for f, o in zip(force, outs):
            text = o.text
            for e in ("<|im_end|>", "<|endoftext|>"):
                text = text.replace(e, "")
            text = text.strip()
            if f and tag == "reason":
                text = force_alpha_answer(text)
            res.append((text, o.finish_reason == "length"))
        return res

    def gen_locate(self, eps: list[Episode], allowed: list[list[int]]) -> list[list[int]]:
        if self.mode == "masked":
            raise ValueError("the masked mode has no <locate>: use an engine locator")
        prompts = [self._locate_prompt(ep) for ep in eps]
        rx = [locate_regex(al, self.proto.k) for al in allowed]
        outs = self.engine.generate(prompts, max_tokens=self.proto.locate_max_tokens, temperature=self.temperature,
                                    top_p=self.top_p, stop=["</locate>"], regex=rx,
                                    seeds=[ep.seed * 7919 + len(ep.steps) for ep in eps])
        return [parse_ids(o.text)[: self.proto.k] for o in outs]

    def release(self, ep: Episode) -> None:
        ep.state.clear()
