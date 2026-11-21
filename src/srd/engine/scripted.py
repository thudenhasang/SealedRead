"""A scripted policy (no model): follows the hops of the sample. Tests the driver, rewards, filters and metrics on
the CPU, and gives the data checks an oracle run ("every sample is solvable through the protocol")."""
from __future__ import annotations

import random

from srd.engine.driver import Episode
from srd.protocol import pasted


class ScriptedPolicy:
    def __init__(self, noise: float = 0.0, wrong: float = 0.0, seed: int = 0):
        self.noise = noise          # chance to add a distractor ID
        self.wrong = wrong          # chance to answer wrongly
        self.rng = random.Random(seed)

    def _next_hop(self, ep: Episode):
        done = set(pasted(ep.steps))
        for h in ep.sample.hops:
            if not set(h.sids) <= done:
                return h
        return None

    def gen_masked(self, eps, tag, force):
        out = []
        for ep, f in zip(eps, force):
            s = ep.sample
            hop = self._next_hop(ep)
            if tag == "reason":
                need = hop is not None and not f and s.gold
                out.append((f"{'Need ' + hop.bridge if need else 'Enough evidence'}. "
                            f"alpha={'query' if need else 'answer'}", False))
            elif tag == "query":
                out.append((f"step towards {hop.bridge if hop else s.question}", False))
            elif tag == "recheck":
                out.append(("The pasted text answers the query. beta=valid_to_think", False))
            elif tag == "answer":
                ans = "wrong answer" if self.rng.random() < self.wrong else s.answers[0]
                out.append((ans, False))
        return out

    def gen_locate(self, eps, allowed):
        out = []
        for ep, al in zip(eps, allowed):
            hop = self._next_hop(ep)
            ids = [i for i in (hop.sids if hop else []) if i in al][:2] or al[:1]
            if self.rng.random() < self.noise:
                extra = [i for i in al if i not in ids and i not in ep.sample.gold]
                if extra:
                    ids.append(self.rng.choice(extra))
            out.append(ids[:3])
        return out

    def release(self, ep):
        pass
