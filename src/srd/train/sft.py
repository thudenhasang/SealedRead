"""SFT trainer for every branch: LoRA (default) or full fine-tuning, FlexAttention masks on GPU (dense on CPU),
chunked cross-entropy on target tokens only, gradient checkpointing, early stopping on validation loss,
resumable checkpoints under a time cap (the keychain-dv conventions: TIME_CAP / TRAIN_DONE markers).

One optimizer step = `batch_traces` traces; each sequence of a trace is one micro-batch (contexts are long); the loss
is the sum of target-token cross-entropies divided by the number of target tokens in the step. With several GPUs
(torchrun) every rank takes its share of the traces and LoRA gradients are all-reduced by hand.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch

from srd.config import Config
from srd.model.lora import apply_lora, load_lora, save_lora
from srd.model.qwen3 import Qwen3Config, Qwen3ForCausalLM, from_pretrained, save_pretrained
from srd.model.run import forward_seq, target_ce
from srd.prompts import Prompts
from srd.tok import load_tok
from srd.train.data import TraceItem, load_traces, sample_index, sequences


def _dist():
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rank = torch.distributed.get_rank() if ws > 1 else 0
    return rank, ws


def load_model(cfg: Config, device) -> Qwen3ForCausalLM:
    dtype = torch.float32 if cfg.train.get("full_ft") else getattr(torch, cfg.model.get("dtype", "bfloat16"))
    if cfg.model.name == "tiny":                          # CPU smoke runs
        torch.manual_seed(cfg.run.seed)
        return Qwen3ForCausalLM(Qwen3Config.tiny(cfg.model.get("vocab", 259))).to(device)
    return from_pretrained(cfg.model.name, dtype=dtype, device=device)


class SFTTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rank, self.world = _dist()
        self.main = self.rank == 0
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available()
                                   else "cpu")
        self.out = Path(cfg.run.out_dir) / cfg.run.name
        self.out.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.branch = cfg.train.branch
        self.attn = cfg.train.get("attn") or ("flex" if self.device.type == "cuda" else "dense")
        self.pad_to = int(cfg.train.get("pad_to", 1024 if self.attn == "flex" else 1))
        self._ntgt: dict[int, int] = {}
        tok_name = "bytes" if cfg.model.name == "tiny" else (cfg.model.get("tokenizer") or cfg.model.name)
        self.tok = load_tok(tok_name)
        self.prompts = Prompts.from_cfg(cfg.get("prompts"))
        samples = sample_index(list(cfg.data.samples))
        self.train_items = load_traces(cfg.data.train, samples, cfg.data.get("limit"))
        self.val_items = load_traces(cfg.data.val, samples, cfg.data.get("val_limit")) if cfg.data.get("val") else []
        self.model = load_model(cfg, self.device)
        self.model.grad_ckpt = bool(cfg.train.get("grad_ckpt", True))
        self.full_ft = bool(cfg.train.get("full_ft", False))
        if self.full_ft:
            self.params = [p for p in self.model.parameters()]
            for p in self.params:
                p.requires_grad_(True)
        else:
            lc = cfg.lora
            self.params = apply_lora(self.model, lc.r, lc.alpha, lc.get("dropout", 0.0), tuple(lc.target_modules))
        if cfg.get("init_from"):
            self._init_from(Path(cfg.init_from))
        self.opt = torch.optim.AdamW(self.params, lr=cfg.train.lr, weight_decay=cfg.train.get("weight_decay", 0.0),
                                     betas=(0.9, 0.999))
        self.steps_per_epoch = max(1, math.ceil(len(self.train_items) / cfg.train.batch_traces))
        self.total_steps = self.steps_per_epoch * cfg.train.epochs
        self.step, self.epoch, self.best_val, self.bad_epochs = 0, 0, float("inf"), 0
        self.order: list[int] = []
        self._resume()
        if self.main:
            n_tgt = sum(self._n_targets(it) for it in self.train_items[:50])
            print(f"branch={self.branch} attn={self.attn} traces={len(self.train_items)} val={len(self.val_items)} "
                  f"steps/epoch={self.steps_per_epoch} target tokens/trace~{n_tgt / max(1, min(50, len(self.train_items))):.0f}",
                  flush=True)

    # ------------------------------------------------------------------ helpers
    def _seqs(self, it: TraceItem):
        return sequences(it, self.tok, self.prompts, self.branch)

    def _n_targets(self, it: TraceItem) -> int:
        key = id(it)
        if key not in self._ntgt:
            self._ntgt[key] = sum(int(sq.target.sum()) for sq in self._seqs(it))
        return self._ntgt[key]

    def _init_from(self, path: Path) -> None:
        """Start from a previous run (S1 checkpoint for the M-step): its best/ (or last) adapter or full weights."""
        run = path if (path / "adapter").exists() or (path / "model").exists() else (
            path / "best" if (path / "best").exists() else sorted((path / "ckpt").glob("step_*"))[-1])
        if (run / "adapter").exists():
            load_lora(self.model, run / "adapter")
        elif (run / "model").exists():
            sd = from_pretrained(str(run / "model"), dtype=self.model.dtype).state_dict()
            self.model.load_state_dict(sd, strict=False)
        if self.main:
            print(f"initialized from {run}", flush=True)

    def lr_at(self, step: int) -> float:
        base = self.cfg.train.get("full_ft_lr", self.cfg.train.lr) if self.full_ft else self.cfg.train.lr
        warm = int(self.cfg.train.get("warmup_steps", 10))
        if step < warm:
            return base * (step + 1) / warm
        frac = (step - warm) / max(1, self.total_steps - warm)
        floor = self.cfg.train.get("min_lr_ratio", 0.1)
        return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    def _loss_of(self, it: TraceItem, scale: float) -> tuple[float, int]:
        tot, n = 0.0, 0
        for seq in self._seqs(it):
            idx = seq.target_index()
            if not len(idx):
                continue
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                h, t = forward_seq(self.model, seq, self.attn, self.pad_to)
                ce = target_ce(self.model, h, t["ids"], torch.as_tensor(idx, device=self.device))
            (ce * scale).backward()
            tot += float(ce.detach())
            n += len(idx)
        return tot, n

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for i, it in enumerate(self.val_items):
            if i % self.world != self.rank:
                continue
            for seq in self._seqs(it):
                idx = seq.target_index()
                if not len(idx):
                    continue
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    h, t = forward_seq(self.model, seq, self.attn, self.pad_to)
                    tot += float(target_ce(self.model, h, t["ids"], torch.as_tensor(idx, device=self.device)))
                n += len(idx)
        if self.world > 1:
            v = torch.tensor([tot, n], device=self.device, dtype=torch.float64)
            torch.distributed.all_reduce(v)
            tot, n = float(v[0]), int(v[1])
        self.model.train()
        return tot / max(1, n)

    # ------------------------------------------------------------------ checkpoints
    def _save_weights(self, d: Path) -> None:
        if self.full_ft:
            save_pretrained(self.model, d / "model")
        else:
            save_lora(self.model, d / "adapter", base_model=self.cfg.model.name)

    def save(self, tag: str | None = None) -> None:
        if not self.main:
            return
        d = self.out / "ckpt" / f"step_{self.step:05d}"
        d.mkdir(parents=True, exist_ok=True)
        self._save_weights(d)
        torch.save(self.opt.state_dict(), d / "optim.pt")
        (d / "state.json").write_text(json.dumps({"step": self.step, "epoch": self.epoch, "best_val": self.best_val,
                                                  "bad_epochs": self.bad_epochs, "order": self.order}))
        dirs = sorted((self.out / "ckpt").glob("step_*"))
        for old in dirs[:-int(self.cfg.run.get("keep", 2))]:
            shutil.rmtree(old, ignore_errors=True)

    def _resume(self) -> None:
        root = self.out / "ckpt"
        dirs = sorted(p for p in root.glob("step_*") if (p / "state.json").exists()) if root.exists() else []
        if not dirs:
            return
        d = dirs[-1]
        st = json.loads((d / "state.json").read_text())
        if (d / "adapter").exists():
            load_lora(self.model, d / "adapter")
        elif (d / "model").exists():
            self.model.load_state_dict(from_pretrained(str(d / "model"), dtype=self.model.dtype).state_dict(),
                                       strict=False)
        self.opt.load_state_dict(torch.load(d / "optim.pt", map_location="cpu"))
        self.step, self.epoch = st["step"], st["epoch"]
        self.best_val, self.bad_epochs, self.order = st["best_val"], st["bad_epochs"], st["order"]
        if self.main:
            print(f"resumed from {d} at step {self.step}", flush=True)

    def log(self, rec: dict) -> None:
        if not self.main:
            return
        rec = {"step": self.step, "epoch": self.epoch, "minutes": round((time.time() - self.t0) / 60, 2), **rec}
        with open(self.out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

    def out_of_time(self) -> bool:
        cap = self.cfg.run.get("max_minutes", 0)
        return bool(cap) and (time.time() - self.t0) / 60 > cap

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        cfg = self.cfg
        self.model.train()
        bt = cfg.train.batch_traces
        while self.epoch < cfg.train.epochs:
            if len(self.order) != len(self.train_items):          # a new epoch (a resumed one keeps its order)
                self.order = list(range(len(self.train_items)))
                random.Random(cfg.run.seed + self.epoch).shuffle(self.order)
            while self.step < (self.epoch + 1) * self.steps_per_epoch:
                if self.out_of_time():
                    self.save()
                    print("TIME_CAP: checkpoint saved, run again to continue", flush=True)
                    return
                t0 = time.time()
                k = self.step % self.steps_per_epoch
                batch = [self.train_items[i] for i in self.order[k * bt:(k + 1) * bt]]
                mine = batch[self.rank::self.world]
                n_tgt = sum(self._n_targets(it) for it in batch)
                for g in self.opt.param_groups:
                    g["lr"] = self.lr_at(self.step)
                self.opt.zero_grad(set_to_none=True)
                tot = 0.0
                for it in mine:
                    loss, _ = self._loss_of(it, 1.0 / max(1, n_tgt))
                    tot += loss
                if self.world > 1:
                    for p in self.params:
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        torch.distributed.all_reduce(p.grad)
                    v = torch.tensor([tot], device=self.device)
                    torch.distributed.all_reduce(v)
                    tot = float(v[0])
                gn = torch.nn.utils.clip_grad_norm_(self.params, cfg.train.get("grad_clip", 1.0))
                self.opt.step()
                self.step += 1
                self.log({"loss": tot / max(1, n_tgt), "lr": self.lr_at(self.step - 1), "grad_norm": float(gn),
                          "target_tokens": n_tgt, "t_step": round(time.time() - t0, 2)})
                if self.step % int(cfg.run.get("save_every", 50)) == 0:
                    self.save()
            self.epoch += 1
            self.order = []
            if self.val_items:
                v = self.validate()
                improved = v < self.best_val - cfg.train.get("min_delta", 0.0)
                self.log({"val_loss": v, "improved": improved})
                if improved:
                    self.best_val, self.bad_epochs = v, 0
                    if self.main:
                        best = self.out / "best"
                        shutil.rmtree(best, ignore_errors=True)
                        self._save_weights(best)
                        (best / "state.json").write_text(json.dumps({"step": self.step, "val_loss": v}))
                else:
                    self.bad_epochs += 1
                    if self.bad_epochs >= cfg.train.get("patience", 1):
                        self.save()
                        print("EARLY_STOP", flush=True)
                        print("TRAIN_DONE", flush=True)
                        return
            self.save()
        if self.main and not (self.out / "best").exists():
            self._save_weights(self.out / "best")
        print("TRAIN_DONE", flush=True)
