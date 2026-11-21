"""Attention backends: SDPA with a causal or explicit boolean mask, and FlexAttention with a BlockMask.

`flex_mask` implements exactly `srd.layout.visible` (tests/test_model.py checks the two agree), so the trainer's
masks and the reference engine's masks come from one rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from srd.layout import C, CPY, GEN, KNOW, LOC, NEVER, PAD, MaskOpts, Seq

_FLEX = None


@dataclass
class Attn:
    mode: str = "causal"                    # causal | dense | flex
    mask: torch.Tensor | None = None        # dense: bool [Tq, Tk] over the keys left after kv_index
    kv_index: torch.Tensor | None = None    # read only these keys (cache + new tokens) before attending
    block_mask: Any = None                  # flex


def flex_fn(compile: bool | None = None):
    """flex_attention, compiled on GPU (the eager version materializes the score matrix: tests only)."""
    global _FLEX
    from torch.nn.attention.flex_attention import flex_attention

    if compile is None:
        compile = torch.cuda.is_available()
    if not compile:
        return flex_attention
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spec: Attn) -> torch.Tensor:
    """q [B, H, Tq, D]; k, v [B, Hkv, Tk, D] -> [B, H, Tq, D]."""
    if spec.kv_index is not None:
        k = k.index_select(2, spec.kv_index)
        v = v.index_select(2, spec.kv_index)
    h, hkv = q.shape[1], k.shape[1]
    if spec.mode == "flex":
        return flex_fn()(q, k, v, block_mask=spec.block_mask, enable_gqa=h != hkv)
    if h != hkv:
        k = k.repeat_interleave(h // hkv, dim=1)
        v = v.repeat_interleave(h // hkv, dim=1)
    if spec.mode == "causal":
        if q.shape[2] != k.shape[2]:
            raise ValueError("causal attention needs as many keys as queries (use a dense mask with a cache)")
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if spec.mode == "dense":
        m = spec.mask
        if m.dim() == 2:
            m = m[None, None]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    raise ValueError(f"unknown attention mode {spec.mode!r}")


# ---------------------------------------------------------------------- whole-sequence masks (training, scoring)
def seq_tensors(seq: Seq, device, pad_to: int = 1) -> dict[str, torch.Tensor]:
    """Per-token tensors of a layout, padded with PAD tokens to a multiple of `pad_to` (stable compiled shapes)."""
    n = len(seq)
    tot = ((n + pad_to - 1) // pad_to) * pad_to
    pad = tot - n

    def t(x, fill):
        a = np.concatenate([x, np.full(pad, fill, dtype=np.int64)]) if pad else x
        return torch.as_tensor(a, dtype=torch.long, device=device)

    last_pos = int(seq.pos.max()) if n else 0
    out = {"ids": t(seq.ids, 0), "kind": t(seq.kind, PAD), "rnd": t(seq.rnd, -1), "open_at": t(seq.open_at, NEVER),
           "stp": t(seq.stp, -2), "qref": t(seq.qref, -3),
           "pos": t(seq.pos, 0) if not pad else torch.as_tensor(
               np.concatenate([seq.pos, last_pos + 1 + np.arange(pad)]), dtype=torch.long, device=device)}
    tgt = np.concatenate([seq.target, np.zeros(pad, bool)]) if pad else seq.target
    out["target"] = torch.as_tensor(tgt, dtype=torch.bool, device=device)
    out["n"] = n
    return out


def dense_mask(seq: Seq, device, pad_to: int = 1) -> torch.Tensor:
    n = len(seq)
    m = seq.mask_dense()
    tot = ((n + pad_to - 1) // pad_to) * pad_to
    if tot > n:
        full = np.zeros((tot, tot), bool)
        full[:n, :n] = m
        full[np.arange(n, tot), np.arange(n, tot)] = True
        m = full
    return torch.as_tensor(m, device=device)


def make_mask_mod(t: dict[str, torch.Tensor], opts: MaskOpts):
    """The mask_mod of the layout tensors `t` (from seq_tensors) under `opts`: the same rule as layout.visible."""
    kind, rnd, open_at, stp, qref = t["kind"], t["rnd"], t["open_at"], t["stp"], t["qref"]
    branch, hide, qonly = opts.branch, opts.hide_locate, opts.locate_history == "query_only"

    def mask_mod(b, h, q, k):
        qk, kk = kind[q], kind[k]
        causal = k <= q
        yes = causal | ~causal
        q_prefix = qk <= C
        q_loc = qk == LOC
        if branch == "nomask":
            c_ok = yes
        elif branch == "daft":
            c_ok = q_loc | (open_at[k] <= q)
        else:
            c_ok = q_loc
        loc_ok = (q_loc & (rnd[k] == rnd[q])) if hide else yes
        if qonly:
            hist = (kk == GEN) | (kk == CPY) | (kk == KNOW)
            other_ok = ~(q_loc & hist) | (stp[k] == qref[q])
        else:
            other_ok = yes
        after = torch.where(kk == LOC, loc_ok, torch.where(kk == C, c_ok, other_ok))
        vis = causal & (q_prefix | after) & (kk != PAD)
        return torch.where(qk == PAD, k == q, vis)

    return mask_mod


def flex_mask(t: dict[str, torch.Tensor], opts: MaskOpts, block_size: int = 128):
    from torch.nn.attention.flex_attention import create_block_mask

    n = t["kind"].shape[0]
    return create_block_mask(make_mask_mod(t, opts), None, None, n, n, device=t["kind"].device,
                             BLOCK_SIZE=block_size)
