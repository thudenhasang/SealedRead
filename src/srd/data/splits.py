"""The whole M1 build: shards -> train / val / reserve / dev / test files (section 4, table 4).

    train    7,000 questions from the train shards, mixed by type (table 4), one random length in 16K-24K each
    val      1,500 questions (same mix, disjoint) for early stopping and the round-2 condition
    reserve  extra questions for teacher top-ups
    dev      1,000 questions from the decision shard s90 at 64K (P0 + the S1 decision point), also 128K and 32K
    test     2,000 other s90 questions at 64K and 128K (the same questions at both), 1,000 of them at 32K;
             their fictional copies at 64K and 128K (the parametric-memory channel, RQ4 counterfactuals)
Type 2 (paraphrase) samples are written with their original question and meta.needs_paraphrase; scripts/paraphrase.py
rewrites them on GPU. A closed-book result file (scripts/closed_book.py) drops train items the base model solves.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from srd.data.build import BuildConfig, PassagePool, build_sample, entity_names
from srd.data.fiction import fictionalize
from srd.data.qa import QAItem, load_shard
from srd.data.sample import Sample, read_jsonl, write_jsonl
from srd.data.synth import enumeration_items, nocontext_question
from srd.teacher import quotas

COMPARISON = ("comparison", "bridge_comparison")


def load_items(shards_dir: str | Path, shards: list[int], datasets: list[str]) -> list[QAItem]:
    out: list[QAItem] = []
    for name in datasets:
        for k in shards:
            p = Path(shards_dir) / f"{name}_s{k}.json"
            if p.exists():
                out.extend(load_shard(p))
    return out


def _len(rng: random.Random, spec) -> int:
    return rng.randint(int(spec[0]), int(spec[1])) if isinstance(spec, (list, tuple)) else int(spec)


class Builder:
    def __init__(self, pool: PassagePool, count: Callable[[str], int], bcfg: BuildConfig, seed: int):
        self.pool, self.count, self.bcfg, self.seed = pool, count, bcfg, seed

    def one(self, item: QAItem, qtype: str, length: int, salt: str) -> Sample:
        rng = random.Random(f"{self.seed}|{item.dataset}|{item.qid}|{length}|{salt}")
        s = build_sample(item, self.pool, rng, length, self.count, self.bcfg, qtype=qtype if qtype != "fictional"
                         else "multihop")
        if qtype == "fictional":
            s = fictionalize(s, entity_names(item), rng)
        if qtype == "paraphrase":
            s.meta["needs_paraphrase"] = True
        return s


def typed_items(items: list[QAItem], n: int, mix: dict[str, float], rng: random.Random, all_items: list[QAItem],
                enum_share: float, start: int = 0) -> list[tuple[QAItem, str]]:
    """Pick n items by type quota; consumes `items` (a shuffled list) from the front."""
    q = quotas(n, mix)
    comp = [it for it in items if it.qtype in COMPARISON]
    regular = [it for it in items if it.qtype not in COMPARISON]
    out: list[tuple[QAItem, str]] = []
    for t in ("multihop", "paraphrase", "fictional"):
        for _ in range(q.get(t, 0)):
            if regular:
                out.append((regular.pop(), t))
    n_ms = q.get("multiseg", 0)
    enum = enumeration_items(all_items, rng, int(round(n_ms * enum_share)))
    for it in enum:
        it.qid = f"{it.qid}-{start}"
        out.append((it, "multiseg"))
    for _ in range(n_ms - len(enum)):
        if comp:
            out.append((comp.pop(), "multiseg"))
    for j in range(q.get("nocontext", 0)):
        out.append((nocontext_question(rng, start + j), "nocontext"))
    used = {id(it) for it, _ in out}
    items[:] = [it for it in items if id(it) not in used]
    return out


def build_all(b: dict, shards_dir: str | Path, out: str | Path, count: Callable[[str], int], seed: int = 0,
              log=print) -> dict:
    out = Path(out)
    rng = random.Random(seed)
    train_items = load_items(shards_dir, list(b["train_shards"]), list(b["datasets"]))
    eval_items = load_items(shards_dir, [b["eval_shard"]], list(b["datasets"]))
    log(f"{len(train_items)} train items, {len(eval_items)} eval items")
    if b.get("closed_book_filter"):
        solved = {r["qid"] for r in read_jsonl(b["closed_book_filter"]) if r.get("correct")}
        before = len(train_items)
        train_items = [it for it in train_items if f"{it.dataset}-{it.qid}" not in solved]
        log(f"closed-book filter: {before - len(train_items)} train items dropped")
    pool = PassagePool(train_items + eval_items)
    bcfg = BuildConfig(b.get("max_seg_tokens", 256), b.get("min_seg_tokens", 32), b.get("hard_distractors", 8))
    builder = Builder(pool, count, bcfg, seed)
    rng.shuffle(train_items)
    rng.shuffle(eval_items)
    manifest: dict = {"seed": seed, "files": {}}
    share = b["train"].get("multiseg_enumeration_share", 0.5)
    for split in ("train", "val", "reserve"):
        spec = b[split]
        picked = typed_items(train_items, spec["n"], b["train"]["mix"], rng, train_items + eval_items, share,
                             start={"train": 0, "val": 100000, "reserve": 200000}[split])
        rows = []
        for i, (it, t) in enumerate(picked):
            rows.append(builder.one(it, t, _len(rng, spec["length"]), split).to_dict())
            if (i + 1) % 500 == 0:
                log(f"{split}: {i + 1}/{len(picked)}")
        manifest["files"][f"{split}.jsonl"] = write_jsonl(out / f"{split}.jsonl", rows)
    dev = eval_items[: b["dev"]["n"]]
    test = eval_items[b["dev"]["n"]: b["dev"]["n"] + b["test"]["n"]]
    for name, items, spec in (("dev", dev, b["dev"]), ("test", test, b["test"])):
        for length, n in spec["lengths"].items():
            rows = [builder.one(it, "multihop", int(length), name).to_dict() for it in items[: int(n)]]
            fn = f"{name}_{int(length) // 1024}k.jsonl"
            manifest["files"][fn] = write_jsonl(out / fn, rows)
            log(f"{fn}: {len(rows)}")
        for length in spec.get("fictional_lengths", []):
            rows = [builder.one(it, "fictional", int(length), name).to_dict() for it in items]
            fn = f"{name}_fict_{int(length) // 1024}k.jsonl"
            manifest["files"][fn] = write_jsonl(out / fn, rows)
            log(f"{fn}: {len(rows)}")
    return manifest
