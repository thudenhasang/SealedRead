"""Mask probes (section 3.6, RQ4): counterfactual prefix forcing on a fixed trajectory.

For a trajectory y generated on C, build C' and force exactly the same token sequence (the same IDs; the engine
pastes the same segments, taken from C'), then compare the distributions at every masked-phase position and at the
final answer.
    invariance  C' changes only text outside the pasted segments (the segment right before a chosen one included),
                with the same token count, so positions after C do not move. SealedRead: invariant up to numerical
                noise (an implementation test: failing means a bug). No-mask control and DA-ft: expected to change;
                the size of the change is the leak they measure.
    dependence  C' changes the answer-bearing value inside a pasted segment: the final answer must follow.
Numerical noise is calibrated by recomputing the same C with a different padding (other kernel tiling).
"""
from __future__ import annotations

import numpy as np
import torch

from srd.data.sample import Sample
from srd.layout import C, GEN, Seq, build_seq
from srd.model.attention import seq_tensors
from srd.model.run import seq_attn
from srd.prompts import Prompts
from srd.protocol import Step, pasted
from srd.text import normalize


@torch.no_grad()
def position_logprobs(model, seq: Seq, rows: np.ndarray, mode: str = "dense", pad_to: int = 1,
                      chunk: int = 256) -> torch.Tensor:
    """log-softmax over the vocabulary at the given storage rows [len(rows), V] (float32, on CPU)."""
    t = seq_tensors(seq, model.device, pad_to=pad_to)
    h = model(t["ids"][None], t["pos"][None], seq_attn(seq, t, mode))[0]
    out = []
    idx = torch.as_tensor(rows, device=h.device)
    for s in range(0, len(rows), chunk):
        out.append(torch.log_softmax(model.logits(h[idx[s:s + chunk]]).float(), -1).cpu())
    return torch.cat(out) if out else torch.zeros(0)


def masked_rows(seq: Seq) -> np.ndarray:
    """Rows whose next-token distribution belongs to the masked phase (they predict a GEN target)."""
    tgt = seq.target_index()
    return (tgt[seq.kind[tgt] == GEN] - 1).astype(np.int64)


def perturb_outside(seq: Seq, keep_sids: set[int], rng: np.random.Generator) -> Seq:
    """Replace every C token outside the kept segments by another token of C (same count, same positions)."""
    sel = (seq.kind == C) & ~np.isin(seq.seg, list(keep_sids))
    pool = seq.ids[seq.kind == C]
    ids = seq.ids.copy()
    ids[sel] = rng.choice(pool, size=int(sel.sum()))
    return Seq(**{**seq.__dict__, "ids": ids})


def kl_rows(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p.exp() * (p - q)).sum(-1)


def invariance(model, s: Sample, steps: list[Step], tok, prompts: Prompts, branch: str = "sealedread",
               mode: str = "dense", pad_to: int = 1, seed: int = 0) -> dict:
    seq = build_seq(s, steps, tok, prompts, view="chain", branch=branch)
    rows = masked_rows(seq)
    rng = np.random.default_rng(seed)
    alt = perturb_outside(seq, set(pasted(steps)), rng)
    a = position_logprobs(model, seq, rows, mode, pad_to)
    b = position_logprobs(model, alt, rows, mode, pad_to)
    noise = position_logprobs(model, seq, rows, mode, pad_to + 128 if mode == "flex" else pad_to)
    kl = kl_rows(a, b)
    return {"qid": s.qid, "branch": branch, "positions": len(rows), "kl_mean": float(kl.mean()) if len(kl) else 0.0,
            "kl_max": float(kl.max()) if len(kl) else 0.0,
            "noise_kl_max": float(kl_rows(a, noise).max()) if len(kl) else 0.0,
            "max_abs_logp": float((a - b).abs().max()) if len(kl) else 0.0}


def answer_rows(seq: Seq, steps: list[Step], tok) -> np.ndarray:
    """Rows that predict the tokens of the final answer text."""
    k = max(i for i, st in enumerate(steps) if st.tag == "answer")
    idx = np.nonzero(seq.stp == k)[0]
    n_open = len(tok.encode("<answer>"))
    n_text = len(tok.encode(steps[k].text))
    return (idx[n_open:n_open + n_text] - 1).astype(np.int64)


def chain_vs_masked(model, s: Sample, steps: list[Step], tok, prompts: Prompts, mode: str = "dense",
                    pad_to: int = 1) -> dict:
    """The masked-phase log-probs of the single chain (C present, masked) against the masked view in the full frame
    (C absent, positions jumping over |C|): the two must agree up to numerical noise. This is what lets S0 config
    (b) run without C. Noise: the chain recomputed with another padding (other kernel tiling)."""
    chain = build_seq(s, steps, tok, prompts, view="chain", branch="sealedread")
    masked = build_seq(s, steps, tok, prompts, view="masked", frame="full")
    rows = masked_rows(chain)
    tgt = torch.as_tensor(chain.ids[rows + 1])
    a = position_logprobs(model, chain, rows, mode, pad_to).gather(1, tgt[:, None])[:, 0]
    b = position_logprobs(model, masked, masked_rows(masked), mode, pad_to).gather(1, tgt[:, None])[:, 0]
    n = position_logprobs(model, chain, rows, mode, 2 * pad_to if pad_to > 1 else 128).gather(1, tgt[:, None])[:, 0]
    d, dn = (a - b).abs().numpy(), (a - n).abs().numpy()
    return {"qid": s.qid, "positions": int(len(rows)), "max": float(d.max()), "median": float(np.median(d)),
            "p99": float(np.percentile(d, 99)), "noise_p99": float(np.percentile(dn, 99))}


def _swap_value(text: str, old: str, new: str) -> str:
    import re

    return re.sub(re.escape(old), new, text, flags=re.I)


def dependence(model, s: Sample, steps: list[Step], tok, prompts: Prompts, new_value: str,
               branch: str = "sealedread", mode: str = "dense", pad_to: int = 1) -> dict | None:
    """Change the answer value inside the pasted segments; log p of the original answer tokens must drop."""
    ans = s.answers[0]
    hit = [g for g in s.segments if g.sid in set(pasted(steps)) and normalize(ans) in normalize(g.text)]
    if not hit:
        return None
    d = s.to_dict()
    for seg in d["segments"]:
        if seg["sid"] in {g.sid for g in hit}:
            seg["text"] = _swap_value(seg["text"], ans, new_value)
    s2 = Sample.from_dict(d)
    seq = build_seq(s, steps, tok, prompts, view="chain", branch=branch)
    seq2 = build_seq(s2, steps, tok, prompts, view="chain", branch=branch)
    ans_rows = answer_rows(seq, steps, tok)
    a = position_logprobs(model, seq, ans_rows, mode, pad_to)
    b = position_logprobs(model, seq2, answer_rows(seq2, steps, tok), mode, pad_to)
    ids = torch.as_tensor(seq.ids[ans_rows + 1])
    la = a.gather(1, ids[:, None])[:, 0].sum()
    lb = b.gather(1, ids[:, None])[:, 0].sum()
    return {"qid": s.qid, "branch": branch, "logp_orig": float(la), "logp_changed": float(lb),
            "drop": float(la - lb)}
