"""Random segment IDs (section 3.1, item 3): drawn without repetition from 1..9999 for every sample, skipping the
numbers that stand alone in the sample's text (years, counts, page numbers), so reasoning never confuses an ID
with a value (S_1961 next to 'born 1961'). Increasing IDs would put long contexts' IDs outside the training range
and tie the ID to the position; random IDs force <locate> to copy labels."""
from __future__ import annotations

import random

from srd.protocol import ID_MAX, ID_MIN
from srd.text import standalone_numbers


def forbidden_ids(texts: list[str]) -> set[int]:
    out: set[int] = set()
    for t in texts:
        out |= standalone_numbers(t)
    return out


def assign_ids(n: int, rng: random.Random, forbidden: set[int] | None = None) -> list[int]:
    pool = [i for i in range(ID_MIN, ID_MAX + 1) if i not in (forbidden or set())]
    if n > len(pool):
        raise ValueError(f"{n} segments but only {len(pool)} free IDs")
    return rng.sample(pool, n)
