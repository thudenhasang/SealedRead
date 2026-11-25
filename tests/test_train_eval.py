"""Training sequences per branch, a tiny SFT run (loss goes down, resume works), ReST-EM selection, metrics,
statistics and gates."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_sample, trace_two_rounds
from srd.data.sample import write_jsonl
from srd.engine.driver import DriverConfig, run_episodes
from srd.engine.scripted import ScriptedPolicy
from srd.eval.gates import p0_decision, rq1
from srd.eval.metrics import aggregate, by_question, episode_record, hop_skip
from srd.eval.stats import noninferiority_n, paired_bootstrap, sample_size, se_paired
from srd.layout import GEN, LOC
from srd.prompts import Prompts
from srd.protocol import ProtoConfig, Step, steps_to_dicts
from srd.rest_em import select
from srd.reward import RewardConfig
from srd.train.data import BRANCHES, TraceItem, sequences


def test_sequences_per_branch(sample, trace, tok, prompts):
    it = TraceItem(sample, trace)
    n = {b: [len(s) for s in sequences(it, tok, prompts, b)] for b in BRANCHES if b != "sealong"}
    assert len(n["sealedread"]) == 2 and len(n["nomask"]) == 1
    masked, loc = sequences(it, tok, prompts, "sealedread")
    assert masked.target[masked.kind == GEN].all() and not (masked.kind == LOC).any()
    assert loc.target.sum() == (loc.kind == LOC).sum()
    rag = sequences(TraceItem(sample, trace), tok, prompts, "rag")
    assert len(rag) == 1
    seal = sequences(TraceItem(sample, None, "<reason>x</reason><answer>Porto</answer>"), tok, prompts, "sealong")
    assert seal[0].target.sum() > 0


def test_stats():
    assert sample_size(0.2, 0.03) in (1744, 1745)          # the proposal: n ≈ 1,744 (1744.2, rounded up)
    assert se_paired(0.2, 1500) == pytest.approx(0.01155, abs=1e-4)
    assert noninferiority_n(0.2, 0.01) > 15000          # margin -1 point at n = 1000 is underpowered (D7)
    rng = np.random.default_rng(0)
    a = {f"q{i}": (f"c{i // 2}", float(rng.random() < 0.6)) for i in range(2000)}
    b = {k: (c, float(rng.random() < 0.5)) for k, (c, _) in a.items()}
    r = paired_bootstrap(a, b, reps=2000)
    assert r["lo"] < r["delta"] < r["hi"] and 0.05 < r["delta"] < 0.15 and r["clusters"] == 1000


def _q(vals, clusters=None):
    return {f"q{i}|64": ((clusters or {}).get(i, f"c{i}"), v) for i, v in enumerate(vals)}


def test_gates():
    pre = {"p0": {"delta_oracle_min": 2.0, "pi_large": 3.0},
           "rq1": {"delta_mask_min": 2.0, "noninferiority_margin": 1.0, "fallback_band": [-1.0, 2.0]}}
    rng = np.random.default_rng(1)
    n = 3000
    d = rng.random(n) < 0.5
    b = d | (rng.random(n) < 0.1)                 # (b) better than (d) by ~5 points
    f = b.copy()
    out = p0_decision(_q(f.astype(float)), _q(d.astype(float)), _q(b.astype(float)), pre)
    assert out["gate_pass"] and not out["pi_large"] and "go to S1" in out["action"]
    assert p0_decision(_q(d.astype(float)), _q(d.astype(float)), None, pre)["gate_pass"] is False
    sr = [_q((d | (rng.random(n) < 0.08)).astype(float)) for _ in range(3)]
    nm = [_q(d.astype(float)) for _ in range(3)]
    r = rq1(sr, nm, [], [], pre)
    assert r["success"]


def _episodes():
    samples = [make_sample(n=12, seed=i, qid=f"q{i}") for i in range(6)]
    return run_episodes(ScriptedPolicy(noise=0.5, wrong=0.3, seed=1), samples, n=4, cfg=DriverConfig())


def test_rest_em_selection():
    eps = _episodes()
    recs, st = select(eps, ProtoConfig(), RewardConfig(), "soft")
    assert st["questions"] == 6 and recs
    per_q = {}
    for r in recs:
        per_q.setdefault(r["qid"], []).append(r)
    for qid, rs in per_q.items():
        assert 1 <= len(rs) <= 2
        assert all(r["recall"] >= 0.5 for r in rs)
        if len(rs) == 2:
            assert (-rs[0]["r_mu"], rs[0]["n_id"]) <= (-rs[1]["r_mu"], rs[1]["n_id"])
    capped, _ = select(eps, ProtoConfig(), RewardConfig(), "soft", cap=3)
    assert len(capped) == 3
    nogate, _ = select(eps, ProtoConfig(), RewardConfig(), "nogate")
    assert len(nogate) >= len(recs)


def test_metrics_and_aggregate():
    eps = _episodes()
    recs = [episode_record(e, ProtoConfig(), RewardConfig()) for e in eps]
    agg = aggregate(recs)
    assert 0 < agg["em"] <= 1 and agg["recall"] == 1.0
    assert agg["hop_skip"] == 0.0                            # the scripted policy follows the hops
    q = by_question(recs)
    assert len(q) == 6 and all(0 <= v <= 1 for _, v in q.values())


def test_hop_skip_detected(sample):
    g1, g2 = [h.sids[0] for h in sample.hops]
    steps = [Step("reason", "alpha=query"), Step("query", "everything"), Step("locate", ids=[g1, g2]),
             Step("knowledge", ids=sorted([g1, g2], key=lambda x: sample.order()[x]), by="engine"),
             Step("recheck", "beta=valid_to_think"), Step("reason", "alpha=answer"), Step("answer", "Porto")]
    assert hop_skip(steps, sample, Prompts()) is True
    assert hop_skip(trace_two_rounds(sample), sample, Prompts()) is False


# ---------------------------------------------------------------------- tiny SFT run
torch = pytest.importorskip("torch")


def _write_run(tmp_path, branch, epochs=6):
    from srd.config import load_config

    samples = [make_sample(n=6, seed=i, gold_pos=(1, 4), qid=f"q{i}") for i in range(3)]
    write_jsonl(tmp_path / "samples.jsonl", [s.to_dict() for s in samples])
    traces = [{"qid": s.qid, "steps": steps_to_dicts(trace_two_rounds(s))} for s in samples]
    write_jsonl(tmp_path / "traces.jsonl", traces)
    cfg_text = f"""
run: {{name: t_{branch}, out_dir: {tmp_path}/runs, seed: 0, save_every: 2, keep: 2}}
model: {{name: tiny}}
lora: {{r: 8, alpha: 16, target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]}}
data: {{samples: [{tmp_path}/samples.jsonl], train: {tmp_path}/traces.jsonl, val: {tmp_path}/traces.jsonl}}
train: {{branch: {branch}, lr: 0.01, epochs: {epochs}, batch_traces: 3, warmup_steps: 1, patience: 10, grad_ckpt: true}}
"""
    (tmp_path / f"{branch}.yaml").write_text(cfg_text)
    return load_config(tmp_path / f"{branch}.yaml")


@pytest.mark.parametrize("branch", ["sealedread", "nomask", "daft"])
def test_tiny_sft_learns_and_resumes(tmp_path, branch):
    from srd.train.sft import SFTTrainer

    cfg = _write_run(tmp_path, branch)
    tr = SFTTrainer(cfg)
    tr.run()
    lines = [json.loads(x) for x in (tmp_path / "runs" / f"t_{branch}" / "metrics.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in lines if "loss" in r]
    assert losses[-1] < losses[0] * 0.8, losses
    assert (tmp_path / "runs" / f"t_{branch}" / "best" / "adapter" / "adapter_config.json").exists()
    cfg.train.epochs = 8                                  # resume: continues from the last checkpoint
    tr2 = SFTTrainer(cfg)
    assert tr2.step == tr.step
    tr2.run()
    assert tr2.step > tr.step


def test_teacher_shards_merge_to_global_quotas():
    """Several teacher processes (each a share of questions and quotas) merge back to the global quotas."""
    from srd.teacher import collect, merge, quotas, shard_quotas, split_total

    assert sum(split_total(3000, i, 4) for i in range(4)) == 3000
    mix = {"multihop": 0.5, "paraphrase": 0.15, "fictional": 0.15, "multiseg": 0.1, "nocontext": 0.1}
    g = quotas(3000, mix)
    assert {t: sum(shard_quotas(3000, mix, i, 4)[t] for i in range(4)) for t in mix} == g
    samples = [make_sample(n=8, seed=i, gold_pos=(1, 5), qid=f"q{i}") for i in range(24)]
    samples += [make_sample(n=8, seed=100 + i, qid=f"n{i}", nocontext=True) for i in range(8)]
    mix2 = {"multihop": 0.75, "nocontext": 0.25}
    parts = []
    for i in range(2):
        recs, st = collect(ScriptedPolicy(), samples[i::2], 12, mix2, DriverConfig(), RewardConfig(),
                           want=shard_quotas(12, mix2, i, 2), log=lambda m: None)
        parts.append(recs)
    out, stats = merge(parts, quotas(12, mix2), 12)
    assert stats["got"] == {"multihop": 9, "nocontext": 3} and len({r["qid"] for r in out}) == 12
