"""CPU rehearsal of the whole pipeline on synthetic shards with a tiny random model (no download, ~1-2 minutes).

Checks the wiring, not the science: build data -> data checks -> scripted "teacher" traces -> SFT for two branches
-> reference engine with the trained adapter -> ReST-EM selection -> evaluation records -> pre-registered gates.

    python scripts/smoke.py [--out runs_smoke]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

NATIONS = ["French", "German", "Italian", "Spanish", "Polish", "Dutch", "Greek", "Swedish"]


def wiki(i: int) -> dict:
    film, person, nation = f"Film Alpha {i}", f"Director Person {i}", NATIONS[i % len(NATIONS)]
    ctx = [[film, [f"{film} is a {1950 + i % 50} drama film directed by {person}.", " It was shot on location."]],
           [person, [f"{person} is a {nation} film director.", f" He was active for {10 + i % 30} years."]]]
    ctx += [[f"Village {i}-{j}", [f"Village {i}-{j} has {100 + j} inhabitants.", " It has a market."]]
            for j in range(8)]
    return {"_id": f"w{i}", "question": f"What is the nationality of the director of {film}?", "answer": nation,
            "type": "compositional", "supporting_facts": [[film, 0], [person, 0]], "context": ctx,
            "evidences": [[film, "director", person], [person, "country of citizenship", nation]]}


def hotpot(i: int) -> dict:
    a, b = f"Harbor Story {i}", f"Lena Marsh {i}"
    ctx = [[a, [f"{a} is a novel written by {b}."]], [b, [f"{b} was born in Riverton {i}."]]]
    ctx += [[f"Lake {i}-{j}", [f"Lake {i}-{j} is a lake of {j + 2} square kilometres."]] for j in range(8)]
    return {"_id": f"h{i}", "question": f"Where was the author of {a} born?", "answer": f"Riverton {i}",
            "type": "bridge", "supporting_facts": [[a, 0], [b, 0]], "context": ctx}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_smoke")
    args = ap.parse_args()
    import torch

    from srd.cli import log
    from srd.config import load_config
    from srd.data import checks
    from srd.data.sample import load_samples, write_jsonl
    from srd.data.splits import build_all
    from srd.engine.driver import DriverConfig, run_episodes
    from srd.engine.ref import RefPolicy
    from srd.engine.scripted import ScriptedPolicy
    from srd.eval import gates
    from srd.eval.metrics import aggregate, by_question
    from srd.eval.runner import evaluate, load_records
    from srd.model.lora import load_lora
    from srd.model.qwen3 import Qwen3Config, Qwen3ForCausalLM
    from srd.prompts import Prompts
    from srd.protocol import ProtoConfig
    from srd.rest_em import select
    from srd.reward import RewardConfig
    from srd.teacher import collect
    from srd.text import approx_tokens
    from srd.tok import ByteTok
    from srd.train.sft import SFTTrainer

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    shards = out / "shards"
    shards.mkdir(parents=True)
    for k, rng in ((0, range(0, 160)), (90, range(1000, 1060))):
        (shards / f"2wikimultihopqa_s{k}.json").write_text(json.dumps([wiki(i) for i in rng]))
        (shards / f"hotpotqa_s{k}.json").write_text(json.dumps([hotpot(i) for i in rng]))
    b = {"train_shards": [0], "eval_shard": 90, "datasets": ["hotpotqa", "2wikimultihopqa"], "max_seg_tokens": 64,
         "min_seg_tokens": 8, "hard_distractors": 3,
         "train": {"n": 40, "length": [600, 900], "mix": {"multihop": 0.5, "paraphrase": 0.0, "fictional": 0.2,
                                                           "multiseg": 0.1, "nocontext": 0.2},
                   "multiseg_enumeration_share": 1.0},
         "val": {"n": 10, "length": [600, 900]}, "reserve": {"n": 10, "length": [600, 900]},
         "dev": {"n": 10, "lengths": {1024: 10}}, "test": {"n": 20, "lengths": {1024: 20, 2048: 20},
                                                         "fictional_lengths": [1024]}}
    data = out / "data"
    man = build_all(b, shards, data, approx_tokens, seed=0, log=log)
    log(f"1/7 data: {man['files']}")
    train = load_samples(data / "train.jsonl")
    for name, r in (("structure", checks.check_structure(train)), ("leaks", checks.check_leaks(train)),
                    ("gold", checks.check_gold_mapping(train)), ("scripted", checks.check_scripted(train))):
        assert r["pass"], (name, r)
    log("2/7 data checks pass")
    proto = ProtoConfig(caps={"reason": 24, "query": 12, "recheck": 16, "answer": 8})
    rcfg = RewardConfig()
    mix = {"multihop": 0.6, "fictional": 0.2, "multiseg": 0.0, "nocontext": 0.2}
    recs, st = collect(ScriptedPolicy(noise=0.3, seed=1), train, 24, mix, DriverConfig(proto=ProtoConfig()), rcfg,
                       log=lambda m: None)
    write_jsonl(out / "traces.jsonl", recs)
    val = load_samples(data / "val.jsonl")
    vrecs, _ = collect(ScriptedPolicy(), val, 6, mix, DriverConfig(), rcfg, log=lambda m: None)
    write_jsonl(out / "traces_val.jsonl", vrecs)
    log(f"3/7 scripted teacher: {len(recs)} traces {st['got']}")
    runs = {}
    for branch in ("sealedread", "nomask"):
        cfg = load_config("configs/debug_tiny.yaml", [
            f"run.name=sft_{branch}", f"run.out_dir={out / 'runs'}", f"train.branch={branch}",
            f"data.samples=[{data / 'train.jsonl'},{data / 'val.jsonl'}]", f"data.train={out / 'traces.jsonl'}",
            f"data.val={out / 'traces_val.jsonl'}"])
        SFTTrainer(cfg).run()
        runs[branch] = out / "runs" / f"sft_{branch}"
    log("4/7 SFT done for sealedread and nomask")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.tiny()).eval()
    load_lora(model, runs["sealedread"] / "best" / "adapter")
    test = load_samples(data / "test_1k.jsonl")[:4]
    pol = RefPolicy(model, ByteTok(), Prompts(), proto=proto, temperature=0.7)
    eps = run_episodes(pol, test, 2, DriverConfig(proto=proto, lenient=True))
    log(f"5/7 reference engine with the trained adapter: {len(eps)} episodes, "
        f"{sum(e.steps[-1].tag == 'answer' for e in eps if e.steps)} reached <answer>")
    seps = run_episodes(ScriptedPolicy(noise=0.5, wrong=0.3, seed=2), train[:20], 4, DriverConfig())
    sel, sst = select(seps, ProtoConfig(), rcfg, "soft")
    log(f"6/7 ReST-EM selection: {sst}")
    tests = load_samples(data / "test_1k.jsonl") + load_samples(data / "test_2k.jsonl")
    for name, pol_ in (("sr", ScriptedPolicy(noise=0.2, wrong=0.1, seed=3)),
                       ("nm", ScriptedPolicy(noise=0.2, wrong=0.3, seed=4))):
        for seed in range(3):
            evaluate(pol_, tests, 1, DriverConfig(), rcfg, out / "eval" / f"{name}_s{seed}.jsonl", seed=seed,
                     log=lambda m: None)
    sr = [by_question(load_records(out / "eval" / f"sr_s{s}.jsonl")) for s in range(3)]
    nm = [by_question(load_records(out / "eval" / f"nm_s{s}.jsonl")) for s in range(3)]
    import yaml

    pre = yaml.safe_load(open("configs/prereg.yaml"))
    r = gates.rq1(sr, nm, [], [], pre)
    agg = aggregate(load_records(out / "eval" / "sr_s0.jsonl"))
    log(f"7/7 evaluation + gates: EM={agg['em']:.2f} recall={agg['recall']:.2f}; RQ1 delta={r['delta_mask']['delta']:.1f} "
        f"[{r['delta_mask']['lo']:.1f}, {r['delta_mask']['hi']:.1f}] -> {r['claim']}")
    print("SMOKE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
