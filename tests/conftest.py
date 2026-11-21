"""Synthetic samples and traces: enough structure to test the protocol, masks, rewards and engines on the CPU."""
from __future__ import annotations

import random

import pytest

from srd.data.sample import Hop, Sample, Segment
from srd.prompts import Prompts
from srd.protocol import Step
from srd.tok import ByteTok

DIRECTOR = "Mara Elling"


def make_sample(n: int = 10, seed: int = 0, gold_pos=(3, 7), qid: str = "q0", nocontext: bool = False) -> Sample:
    rng = random.Random(seed)
    sids = rng.sample(range(100, 9999), n)
    segs = []
    for i in range(n):
        segs.append(Segment(sids[i], f"Filler {i}", "Introduction",
                            f"Filler {i} is a village with {100 + i} houses near the river.", doc=f"d{i}"))
    if not nocontext:
        a, b = gold_pos
        segs[a] = Segment(sids[a], "The Silent Harbor", "Introduction",
                          f"The Silent Harbor is a 1998 film by {DIRECTOR}.", doc="film", gold=True)
        segs[b] = Segment(sids[b], DIRECTOR, "Introduction",
                          f"{DIRECTOR} (born 1961 in Porto) is a film director.", doc="person", gold=True)
        hops = [Hop([sids[a]], DIRECTOR), Hop([sids[b]], "Porto")]
        return Sample(qid, "2wikimultihopqa", "multihop", "Where was the director of The Silent Harbor born?",
                      ["Porto"], "open", segs, hops, cluster=qid, length=0)
    return Sample(qid, "synthetic", "nocontext", "What is 17 plus 25?", ["42"], "open", segs, [], cluster=qid)


def gold_ids(s: Sample) -> list[int]:
    return s.gold


def trace_two_rounds(s: Sample) -> list[Step]:
    """The example of the proposal: two rounds, then answer."""
    g1, g2 = [h.sids[0] for h in s.hops]
    noise = next(x.sid for x in s.segments if not x.gold)
    order = s.order()
    first = [g1, noise]
    return [
        Step("reason", "Need (1) the director, (2) where they were born. alpha=query"),
        Step("query", "director of The Silent Harbor"),
        Step("locate", ids=first),
        Step("knowledge", ids=sorted(first, key=lambda x: order[x]), by="engine"),
        Step("recheck", f"S_{g1}: the director is {DIRECTOR}. beta=valid_to_think"),
        Step("reason", f"Missing the birthplace of {DIRECTOR}. alpha=query"),
        Step("query", f"birthplace of {DIRECTOR}"),
        Step("locate", ids=[g2]),
        Step("knowledge", ids=[g2], by="engine"),
        Step("recheck", "Born in Porto. beta=valid_to_think"),
        Step("reason", f"Check: S_{g1} -> {DIRECTOR}; S_{g2} -> Porto. alpha=answer"),
        Step("answer", "Porto"),
    ]


@pytest.fixture
def tok():
    return ByteTok()


@pytest.fixture
def prompts():
    return Prompts()


@pytest.fixture
def sample():
    return make_sample()


@pytest.fixture
def trace(sample):
    return trace_two_rounds(sample)
