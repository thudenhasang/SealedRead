"""Fictional entity substitution (section 4.2): rename the question's entities consistently across the whole sample
(question, answers, every segment including distractors). Names are built from syllables so their length and
capitalization match the originals (token statistics close to real names). Closes the parametric-memory channel."""
from __future__ import annotations

import random
import re

from srd.data.sample import Sample
from srd.text import is_weak_span

SYLLABLES = ("ka ri mo len dar vi sto nel ber tha quin mar lo sen av or el bran dov fel gar hal ist jor kel lun mir "
             "nor pell quor ros sil tor ul ven wyn yar zel ash bel cor den eth fen gil hes ior jan kor lis mav "
             "nim osk pav rel sav tev ura vos wil xan yev zor").split()
FUNCTION = {"The", "A", "An", "Of", "In", "On", "And", "For", "To", "At", "By", "De", "La", "Le", "Von", "Van", "Der",
            "Du", "Di", "Da", "St", "Saint", "Mr", "Mrs", "Dr", "Sir", "II", "III", "IV", "Jr", "Sr"}


def fake_word(rng: random.Random, n_chars: int) -> str:
    w = ""
    while len(w) < max(3, n_chars - 1):
        w += rng.choice(SYLLABLES)
    return w[: max(3, n_chars + 1)].capitalize()


def fake_name(rng: random.Random, name: str, word_map: dict[str, str]) -> str:
    out = []
    for w in name.split():
        core = re.sub(r"[^\w]", "", w)
        if core and core[0].isupper() and core.isalpha() and core not in FUNCTION:
            if core not in word_map:
                word_map[core] = fake_word(rng, len(core))
            out.append(w.replace(core, word_map[core]))
        else:
            out.append(w)
    return " ".join(out)


def is_name(s: str) -> bool:
    s = s.strip()
    return bool(s) and s[0].isupper() and not is_weak_span(s) and any(c.isalpha() for c in s) and len(s) <= 80


def fictionalize(s: Sample, entities: list[str], rng: random.Random) -> Sample:
    """A renamed copy of the sample. `entities`: names to replace (gold titles, bridges, name-like answers)."""
    ents = sorted({e.strip() for e in entities if is_name(e)}, key=len, reverse=True)
    corpus = " ".join([s.question] + [g.title + " " + g.text for g in s.segments]).lower()
    word_map: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for e in ents:
        for _ in range(20):
            new = fake_name(rng, e, word_map)
            if new.lower() not in corpus:
                break
            for w in e.split():                 # collision: draw new words
                word_map.pop(re.sub(r"[^\w]", "", w), None)
        mapping[e] = new
    # surnames alone ("Elling" for "Mara Elling"): the last name-like word of multi-word person-like names
    for e, new in list(mapping.items()):
        ws, nw = e.split(), new.split()
        if 2 <= len(ws) <= 3 and len(ws) == len(nw) and len(ws[-1]) >= 4 and ws[-1] not in FUNCTION:
            mapping.setdefault(ws[-1], nw[-1])
    if not mapping:
        return s
    rx = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r")(?!\w)")

    def sub(t: str) -> str:
        return rx.sub(lambda m: mapping[m.group(1)], t)

    d = s.to_dict()
    d["question"] = sub(d["question"])
    d["answers"] = [sub(a) for a in d["answers"]]
    for seg in d["segments"]:
        seg["title"], seg["text"] = sub(seg["title"]), sub(seg["text"])
    for h in d["hops"]:
        h["bridge"] = sub(h["bridge"])
    d["qid"] = s.qid + "-fict"
    d["qtype"] = "fictional"
    d["meta"] = {**d.get("meta", {}), "fiction_map": mapping}
    return Sample.from_dict(d)
