"""S2 sampling and filtering of one ReST-EM round (section 5.4); the M-step is train_sft.py.

    python scripts/rest_em.py configs/rest_em.yaml                       # sample N=4 per question, keep all episodes
    python scripts/rest_em.py configs/rest_em.yaml --select-only rest_em.variant=hard     # RQ3: re-filter, no sampling
Outputs (rest_em.out): episodes.jsonl (every sample, for RQ3), selected_<variant>.jsonl (the M-step data), stats.
Then:  python scripts/train_sft.py configs/sft.yaml train.branch=<branch> init_from=<S1 run> \
           data.train=<out>/selected_soft.jsonl run.name=<rest run>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import config_from_argv, log, shard_args, shard_suffix, to_plain, write_json  # noqa: E402
from srd.data.sample import load_samples, read_jsonl, write_jsonl  # noqa: E402
from srd.engine.driver import DriverConfig, Episode, run_episodes  # noqa: E402
from srd.engine.factory import make_policy  # noqa: E402
from srd.engine.retriever import make_retriever  # noqa: E402
from srd.eval.runner import shard  # noqa: E402
from srd.prompts import Prompts  # noqa: E402
from srd.protocol import ProtoConfig, steps_from_dicts, steps_to_dicts  # noqa: E402
from srd.rest_em import select  # noqa: E402
from srd.reward import RewardConfig  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def adapter_of(run: str) -> str | None:
    p = Path(run)
    for c in (p / "best" / "adapter", p / "adapter"):
        if (c / "adapter_config.json").exists():
            return str(c)
    return None


def main() -> int:
    cfg, flags = config_from_argv()
    r = to_plain(cfg.rest_em)
    out = Path(r["out"])
    proto = ProtoConfig.from_cfg(to_plain(cfg.protocol))
    rcfg = RewardConfig.from_cfg(to_plain(cfg.reward))
    prompts = Prompts.from_cfg(to_plain(cfg.get("prompts")))
    samples = load_samples(r["samples"])
    by = {s.qid: s for s in samples}
    ep_file = out / "episodes.jsonl"
    if "--select-only" not in flags:
        ecfg = {**to_plain(cfg.engine), "temperature": r["temperature"], "top_p": r.get("top_p", 1.0)}
        ecfg["adapter"] = ecfg.get("adapter") or adapter_of(r["init_from"])
        pol, tok, locator = make_policy(ecfg, r["branch"], proto, prompts)
        retr = make_retriever(to_plain(cfg.get("retriever"))) if locator == "retriever" else None
        drv = DriverConfig(proto=proto, locator=locator)
        i, n = shard_args()
        mine = shard(samples, i, n)
        path = shard_suffix(ep_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(x)["qid"] for x in open(path)} if path.exists() else set()
        todo = [s for s in mine if s.qid not in done]
        for b in range(0, len(todo), 128):
            eps = run_episodes(pol, todo[b:b + 128], r["n"], drv, retriever=retr, seed=cfg.run.seed * 7 + b)
            with open(path, "a", encoding="utf-8") as f:
                for ep in eps:
                    f.write(json.dumps({"qid": ep.sample.qid, "j": ep.sample_idx, "error": ep.error,
                                        "steps": steps_to_dicts(ep.steps)}) + "\n")
            log(f"sampled {min(b + 128, len(todo))}/{len(todo)} questions")
        if n > 1:
            print("REST_SAMPLE_SHARD_DONE")
            return 0
    files = sorted(out.glob("episodes*.jsonl"))
    eps = []
    for f in files:
        for d in read_jsonl(f):
            ep = Episode(by[d["qid"]], steps_from_dicts(d["steps"]), error=d.get("error"), sample_idx=d["j"])
            eps.append(ep)
    stok = load_tok("bytes" if cfg.model.name == "tiny" else (cfg.model.get("tokenizer") or cfg.model.name))
    recs, stats = select(eps, proto, rcfg, r["variant"], r["n"], r["keep_per_question"], r["cap"], r["recall_min"],
                         cfg.run.seed, token_count=lambda t: len(stok.encode(t)))
    write_jsonl(out / f"selected_{r['variant']}.jsonl", recs)
    write_json(out / f"selected_{r['variant']}.stats.json", stats)
    log(f"selected {len(recs)} samples: {stats}")
    print("REST_SELECT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
