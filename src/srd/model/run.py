"""Whole-sequence forward passes over a layout: training loss, target log-probabilities, scoring for probes."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from srd.layout import Seq
from srd.model.attention import Attn, dense_mask, flex_mask, seq_tensors


def seq_attn(seq: Seq, t: dict, mode: str) -> Attn:
    if mode == "flex":
        return Attn("flex", block_mask=flex_mask(t, seq.opts))
    if mode == "dense":
        return Attn("dense", mask=dense_mask(seq, t["ids"].device, pad_to=len(t["ids"])))
    raise ValueError(mode)


def forward_seq(model, seq: Seq, mode: str = "dense", pad_to: int = 1) -> tuple[torch.Tensor, dict]:
    """Hidden states [T_pad, H] of the whole layout under its own mask and positions."""
    t = seq_tensors(seq, model.device, pad_to=pad_to)
    h = model(t["ids"][None], t["pos"][None], seq_attn(seq, t, mode))[0]
    return h, t


def _ce_piece(h, w_head, tgt, reduction):
    logits = F.linear(h, w_head).float()
    return F.cross_entropy(logits, tgt, reduction=reduction)


def target_ce(model, hidden: torch.Tensor, ids: torch.Tensor, target_idx: torch.Tensor, chunk: int = 2048,
              reduction: str = "sum") -> torch.Tensor:
    """Cross-entropy of the target tokens, from the hidden state just before each (chunked, logits recomputed in
    backward, so [tokens x vocab] float logits never live in memory at once)."""
    rows = hidden.index_select(0, target_idx - 1)
    tgt = ids.index_select(0, target_idx)
    w = model.lm_head.weight
    parts = []
    for s in range(0, rows.shape[0], chunk):
        h, y = rows[s:s + chunk], tgt[s:s + chunk]
        if torch.is_grad_enabled():
            parts.append(checkpoint(_ce_piece, h, w, y, reduction, use_reentrant=False))
        else:
            parts.append(_ce_piece(h, w, y, reduction))
    if not parts:
        return hidden.new_zeros(())
    return torch.stack(parts).sum() if reduction == "sum" else torch.cat(parts)


@torch.no_grad()
def target_logprobs(model, seq: Seq, mode: str = "dense") -> torch.Tensor:
    """log p of every target token of the layout (in storage order of the targets)."""
    h, t = forward_seq(model, seq, mode)
    idx = torch.as_tensor(seq.target_index(), device=h.device)
    return -target_ce(model, h, t["ids"], idx, reduction="none")


@torch.no_grad()
def token_logprobs(model, seq: Seq, index, mode: str = "dense") -> torch.Tensor:
    """log p of the tokens at the given storage indices (each from the previous token)."""
    h, t = forward_seq(model, seq, mode)
    idx = torch.as_tensor(index, device=h.device)
    return -target_ce(model, h, t["ids"], idx, reduction="none")
