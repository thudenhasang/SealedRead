"""Figure 1b of the proposal, cell by cell, plus positions and views."""
from __future__ import annotations

import numpy as np
import pytest

from srd.layout import (A, C, CPY, GEN, KNOW, LOC, Q, QP, TP, MaskOpts, build_seq)

KINDS = {"A": A, "Q": Q, "C": C, "QP": QP, "GEN": GEN, "LOC": LOC, "CPY": CPY, "KNOW": KNOW}


def _first(seq, kind, rnd=None):
    idx = np.nonzero((seq.kind == kind) & ((seq.rnd == rnd) if rnd is not None else True))[0]
    return idx


def _sees(seq, M, qkind, kkind, q_rnd=None, k_rnd=None):
    """Whether the tokens of qkind (round q_rnd) see *any* / *all* earlier tokens of kkind (round k_rnd)."""
    qs = _first(seq, qkind, q_rnd)
    ks = _first(seq, kkind, k_rnd)
    sub = M[np.ix_(qs, ks)]
    causal = ks[None, :] <= qs[:, None]
    vals = sub[causal]
    return (bool(vals.all()) if vals.size else None), (bool(vals.any()) if vals.size else None)


@pytest.fixture
def chain(sample, trace, tok, prompts):
    seq = build_seq(sample, trace, tok, prompts, view="chain", branch="sealedread")
    return seq, seq.mask_dense()


def test_figure_1b_sealedread(chain):
    seq, M = chain
    # C (prefill) sees A, Q, C
    assert _sees(seq, M, C, A)[0] and _sees(seq, M, C, Q)[0] and _sees(seq, M, C, C)[0]
    # Q', T', masked tokens, pasted text, copies: never C, never original LOC
    for qk in (QP, TP, GEN, KNOW, CPY):
        assert _sees(seq, M, qk, C)[1] is False, qk
        assert _sees(seq, M, qk, A)[0] and _sees(seq, M, qk, Q)[0]
    for qk in (GEN, KNOW, CPY):
        assert _sees(seq, M, qk, LOC)[1] in (False, None)
    # masked tokens see Q', copies, pasted text
    assert _sees(seq, M, GEN, QP)[0] and _sees(seq, M, GEN, CPY)[0] and _sees(seq, M, GEN, KNOW)[0]
    # <locate> of round 2: sees C, history, the copy of round 1, its own round, not round 1's original
    assert _sees(seq, M, LOC, C, q_rnd=2)[0]
    assert _sees(seq, M, LOC, GEN, q_rnd=2)[0] and _sees(seq, M, LOC, KNOW, q_rnd=2)[0]
    assert _sees(seq, M, LOC, CPY, q_rnd=2, k_rnd=1)[0]
    assert _sees(seq, M, LOC, LOC, q_rnd=2, k_rnd=1)[1] is False
    assert _sees(seq, M, LOC, LOC, q_rnd=2, k_rnd=2)[0]


def test_copy_reuses_positions_and_length(chain):
    seq, _ = chain
    for r in (1, 2):
        lo, cp = _first(seq, LOC, r), _first(seq, CPY, r)
        assert len(lo) == len(cp)
        assert np.array_equal(seq.pos[lo], seq.pos[cp])
        kn = _first(seq, KNOW, r)
        assert seq.pos[kn[0]] == seq.pos[lo[-1]] + 1      # pasted text continues after the generation position


def test_copy_is_sorted(chain, tok, sample, trace):
    seq, _ = chain
    cp = tok.decode(seq.ids[_first(seq, CPY, 1)].tolist())
    ids = sorted(trace[2].ids)
    assert cp == "<locate>" + " ".join(f"[[S_{i}]]" for i in ids) + "</locate>"


def test_targets(chain):
    seq, _ = chain
    t = seq.target
    assert t[seq.kind == GEN].all() and t[seq.kind == LOC].all()
    for k in (A, Q, C, QP, TP, CPY, KNOW):
        assert not t[seq.kind == k].any()


def test_nomask_sees_c_and_still_hides_locate(sample, trace, tok, prompts):
    seq = build_seq(sample, trace, tok, prompts, view="chain", branch="nomask")
    M = seq.mask_dense()
    for qk in (QP, TP, GEN, KNOW, CPY):
        assert _sees(seq, M, qk, C)[0]
    assert _sees(seq, M, GEN, LOC)[1] is False


def test_daft_opens_only_chosen_segments(sample, trace, tok, prompts):
    seq = build_seq(sample, trace, tok, prompts, view="chain", branch="daft")
    M = seq.mask_dense()
    gen = _first(seq, GEN)
    last = gen[-1]
    opened = set(trace[2].ids) | set(trace[7].ids)
    for sid_seg in {s.sid for s in sample.segments}:
        cols = np.nonzero((seq.kind == C) & (seq.seg == sid_seg))[0]
        assert M[last, cols].all() == (sid_seg in opened)
    first_gen = gen[0]                    # before any locate: nothing of C
    assert not M[first_gen, seq.kind == C].any()
    # the pasted block carries no text in daft
    assert (seq.kind == KNOW).sum() == 2 * len(tok.encode("\n<knowledge></knowledge>\n"))


def test_masked_view_full_vs_compact(sample, trace, tok, prompts):
    chain = build_seq(sample, trace, tok, prompts, view="chain")
    full = build_seq(sample, trace, tok, prompts, view="masked", frame="full")
    comp = build_seq(sample, trace, tok, prompts, view="masked", frame="compact")
    n_c = int((chain.kind == C).sum())
    assert (full.kind == C).sum() == 0 and (full.kind == LOC).sum() == 0
    # same tokens as the chain without C and without the originals; same positions in the full frame
    keep = (chain.kind != C) & (chain.kind != LOC)
    assert np.array_equal(full.ids, chain.ids[keep])
    assert np.array_equal(full.pos, chain.pos[keep])
    assert np.array_equal(comp.pos[comp.kind >= QP], full.pos[full.kind >= QP] - n_c)
    assert full.target.sum() == chain.target[chain.kind == GEN].sum()


def test_locate_view(sample, trace, tok, prompts):
    seq = build_seq(sample, trace, tok, prompts, view="locate")
    M = seq.mask_dense()
    assert _sees(seq, M, GEN, C)[0]
    assert seq.target[seq.kind == LOC].all() and not seq.target[seq.kind == GEN].any()
    ids = seq.prompt_ids()
    assert len(ids) == len(seq) - (seq.kind == LOC).sum()


def test_prompt_ids_refuses_skipped_c(sample, trace, tok, prompts):
    seq = build_seq(sample, trace, tok, prompts, view="masked", frame="full")
    with pytest.raises(ValueError):
        seq.prompt_ids()


def test_ablation_options(sample, trace, tok, prompts):
    seq = build_seq(sample, trace, tok, prompts, view="chain", opts=MaskOpts(hide_locate=False))
    M = seq.mask_dense()
    assert _sees(seq, M, GEN, LOC, k_rnd=1)[1]
    seq = build_seq(sample, trace, tok, prompts, view="chain", opts=MaskOpts(locate_history="query_only"))
    M = seq.mask_dense()
    lo2 = _first(seq, LOC, 2)
    q2 = np.nonzero(seq.stp == 6)[0]                 # the second <query>
    r1 = np.nonzero(seq.stp == 0)[0]                 # the first <reason>
    assert M[np.ix_(lo2, q2)].any() and not M[np.ix_(lo2, r1)].any()


def test_every_row_sees_something(chain):
    seq, M = chain
    assert M.any(axis=1).all()
    assert np.array_equal(np.diag(M), np.ones(len(seq), bool))


def test_empty_prefix_override_is_empty_string():
    """`prompts.assistant_prefix=` on the command line reads as None: it must mean no prefix, not a crash."""
    from srd.prompts import Prompts

    p = Prompts.from_cfg({"assistant_prefix": None})
    assert p.assistant_start() == "<|im_start|>assistant\n"


def test_instruction_has_no_fill_in_templates():
    """The 0.6B rehearsal copied '<reason> what you know so far ...' word for word: tag descriptions stay outside the
    tags, and the worked example's masked steps end with the canonical alpha / beta (D28)."""
    import re

    from srd.prompts import INSTRUCTION
    from srd.protocol import parse_alpha, parse_beta

    for tag, text in re.findall(r"<(reason|recheck)>(.*?)</\1>", INSTRUCTION, re.S):
        assert (parse_alpha if tag == "reason" else parse_beta)(text) is not None, text
    assert "<reason> what you know" not in INSTRUCTION
