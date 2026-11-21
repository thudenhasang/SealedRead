"""M0-style cost calibration on the target GPU (plan, phase 5): engine throughput, SFT step time, memory, and
that a LoRA adapter loaded into vLLM changes the outputs. The numbers replace the estimates of tables 7-8.

    python scripts/cost_probe.py configs/eval.yaml --samples /content/data/srd/dev_64k.jsonl --n 20 --part engine
    python scripts/cost_probe.py configs/sft.yaml --part sft --steps 5
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from srd.cli import config_from_argv, log, to_plain, write_json  # noqa: E402


def _flag(flags, name, default=None):
    return flags[flags.index(name) + 1] if name in flags else default


def engine_part(cfg, flags) -> dict:
    from srd.data.sample import load_samples
    from srd.engine.driver import DriverConfig, run_episodes
    from srd.engine.factory import make_vllm_engine
    from srd.engine.vllm_policy import VLLMPolicy
    from srd.prompts import Prompts
    from srd.protocol import ProtoConfig
    from srd.tok import load_tok

    ecfg = to_plain(cfg.engine)
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    proto = ProtoConfig.from_cfg(to_plain(cfg.protocol))
    ss = load_samples(_flag(flags, "--samples"), int(_flag(flags, "--n", 20)))
    eng = make_vllm_engine(ecfg, enable_lora=True)
    res = {"length": ss[0].length if ss else None}
    for mode in ("two_stream", "nomask"):
        pol = VLLMPolicy(eng, tok, Prompts(), mode, proto, temperature=0.7)
        t0 = time.time()
        eps = run_episodes(pol, ss, 4, DriverConfig(proto=proto, lenient=True))
        dt = time.time() - t0
        res[mode] = {"episodes": len(eps), "s_per_episode": dt / max(1, len(eps)),
                     "rounds": sum(sum(1 for s in e.steps if s.tag == "locate") for e in eps) / max(1, len(eps))}
        log(f"{mode}: {res[mode]}")
    # LoRA in vLLM really changes the outputs (a no-op LoRA was reported for some models)
    import tempfile

    from srd.engine.factory import load_ref_model
    from srd.model.lora import apply_lora, save_lora

    model = load_ref_model({**ecfg, "dtype": "bfloat16"}, device="cpu")
    params = apply_lora(model, 8, 16)
    with torch.no_grad():
        for p in params:
            p.normal_(std=0.05)
    d = tempfile.mkdtemp()
    save_lora(model, d, ecfg["model"])
    prompt = [tok.encode("<|im_start|>user\nName three colours.<|im_end|>\n<|im_start|>assistant\n")]
    eng.set_adapter(None)
    a = eng.generate(prompt, 24)[0].text
    eng.set_adapter(d)
    b = eng.generate(prompt, 24)[0].text
    res["lora_changes_output"] = a != b
    log(f"LoRA changes the output: {a != b}")
    return res


def sft_part(cfg, flags) -> dict:
    import numpy as np

    from srd.train.sft import SFTTrainer

    steps = int(_flag(flags, "--steps", 5))
    cfg.train.epochs = 1
    cfg.data.limit = steps * cfg.train.batch_traces
    tr = SFTTrainer(cfg)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        t0 = time.time()
        k = tr.step
        bt = cfg.train.batch_traces
        batch = tr.train_items[k * bt:(k + 1) * bt]
        n_tgt = sum(tr._n_targets(it) for it in batch)
        tr.opt.zero_grad(set_to_none=True)
        for it in batch:
            tr._loss_of(it, 1.0 / max(1, n_tgt))
        tr.opt.step()
        tr.step += 1
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    lens = [len(sq) for it in tr.train_items[:20] for sq in tr._seqs(it)]
    return {"s_per_step": float(np.mean(times[1:] or times)), "traces_per_step": cfg.train.batch_traces,
            "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "mean_seq_tokens": float(np.mean(lens)),
            "branch": cfg.train.branch, "full_ft": bool(cfg.train.get("full_ft"))}


def main() -> int:
    cfg, flags = config_from_argv()
    part = _flag(flags, "--part", "engine")
    res = engine_part(cfg, flags) if part == "engine" else sft_part(cfg, flags)
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    write_json(Path(cfg.run.out_dir) / f"cost_probe_{part}.json", res)
    log(str(res))
    print("COST_PROBE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
