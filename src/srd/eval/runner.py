"""Run a policy over samples and write one record per episode (resumable, sharded)."""
from __future__ import annotations

import json
from pathlib import Path

from srd.data.sample import Sample
from srd.engine.driver import DriverConfig, run_episodes
from srd.eval.metrics import episode_record
from srd.prompts import Prompts
from srd.protocol import steps_to_dicts
from srd.reward import RewardConfig


def shard(samples: list[Sample], i: int, n: int) -> list[Sample]:
    return samples[i::n] if n > 1 else samples


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.add(f"{r['qid']}|{r['length']}|{r['j']}")
    return out


def evaluate(policy, samples: list[Sample], n: int, drv: DriverConfig, rcfg: RewardConfig, out_path: str | Path,
             prompts: Prompts | None = None, retriever=None, token_count=None, seed: int = 0, batch: int = 128,
             keep_steps: bool = True, extra: dict | None = None, log=print) -> int:
    """Appends records to out_path; skips samples whose n episodes are already there."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = done_keys(out_path)
    todo = [s for s in samples if any(f"{s.qid}|{s.length}|{j}" not in have for j in range(n))]
    proto = drv.effective_proto()
    written = 0
    for b in range(0, len(todo), batch):
        chunk = todo[b:b + batch]
        eps = run_episodes(policy, chunk, n, drv, retriever=retriever, seed=seed + b)
        with open(out_path, "a", encoding="utf-8") as f:
            for ep in eps:
                rec = episode_record(ep, proto, rcfg, prompts, token_count, extra)
                if keep_steps:
                    rec["steps"] = steps_to_dicts(ep.steps)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        log(f"eval: {min(b + batch, len(todo))}/{len(todo)} samples -> {out_path.name}")
    return written


def load_records(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]
