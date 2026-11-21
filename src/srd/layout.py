"""Token layout of a trace, and the attention rule of every branch: the one source of truth for masks.

Every engine (reference engine, vLLM prompts, trainer, probes) builds its token sequences here and takes its
attention masks from `visible()` (dense, numpy) or `srd.model.attention.flex_mask` (FlexAttention), which
implements exactly the same rule. Equation (1) and Figure 1b of the proposal.

Storage order (= KV order) of a single chain:
    [A][Q][C][Q'][T'] [GEN ...] [LOC_1][CPY_1][KNOW_1] [GEN ...] [LOC_2][CPY_2][KNOW_2] ... [GEN ... <answer> eos]
Kinds:
    A, Q, C, QP (Q'), TP (T')   the prompt
    GEN    tokens the model writes in the masked phase (<reason> <query> <recheck> <answer>)
    LOC    an original <locate> block (the only post-C tokens that see C); hidden once the tag closes
    CPY    its copy: same length and position IDs, IDs sorted by value, encoded with C masked
    KNOW   the pasted segments (engine), encoded with C masked
Position IDs: C keeps its positions; everything after C continues, except that a copy reuses the positions of its
original. With `frame="compact"` (appendix A) the masked view starts right after Q, as if C did not exist.

Views (which tokens exist, which are trained):
    chain    the single chain above, mask by branch (sealedread | nomask | daft); targets GEN + LOC
    masked   no C and no original LOC (copies only): what the masked phase reads; targets GEN.
             frame "full" skips |C| positions (config (b) of S0); "compact" does not (appendix A, two-stream engine)
    locate   the chain where every token sees C (the locate stream of the two-stream engine); targets LOC only
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from srd.data.sample import Sample
from srd.prompts import Prompts
from srd.protocol import MASKED_TAGS, Step, locate_pieces
from srd.tok import Tok

A, Q, C, QP, TP, GEN, LOC, CPY, KNOW, PAD = range(10)
KIND_NAMES = ("A", "Q", "C", "QP", "TP", "GEN", "LOC", "CPY", "KNOW", "PAD")
NEVER = 1 << 60
BRANCHES = ("sealedread", "nomask", "daft")
VIEWS = ("chain", "masked", "locate")


@dataclass
class MaskOpts:
    branch: str = "sealedread"          # sealedread | nomask | daft  (table 10)
    hide_locate: bool = True            # KV of an original <locate> is seen only by its own round (rule of 3.3)
    locate_history: str = "all"         # "all" | "query_only" (ablation: <locate> reads only its own <query>)

    def __post_init__(self):
        if self.branch not in BRANCHES:
            raise ValueError(f"unknown branch {self.branch!r}")
        if self.locate_history not in ("all", "query_only"):
            raise ValueError(f"unknown locate_history {self.locate_history!r}")


def visible(q_idx, q_kind, q_rnd, q_qref, k_idx, k_kind, k_rnd, k_open, k_stp, opts: MaskOpts) -> np.ndarray:
    """Boolean [nq, nk]: may query token q attend key token k (indices are storage indices)."""
    qi, qk, qr, qq = (np.asarray(x)[:, None] for x in (q_idx, q_kind, q_rnd, q_qref))
    ki, kk, kr, ko, ks = (np.asarray(x)[None, :] for x in (k_idx, k_kind, k_rnd, k_open, k_stp))
    causal = ki <= qi
    q_prefix = qk <= C                                   # A, Q, C: plain causal prefill
    q_loc = qk == LOC
    if opts.branch == "nomask":
        c_ok = np.ones(np.broadcast_shapes(qi.shape, ki.shape), bool)
    elif opts.branch == "daft":
        c_ok = q_loc | (ko <= qi)                        # segments opened by an earlier <locate>
    else:
        c_ok = np.broadcast_to(q_loc, np.broadcast_shapes(qi.shape, ki.shape))
    loc_ok = (q_loc & (kr == qr)) if opts.hide_locate else np.ones_like(causal)
    if opts.locate_history == "query_only":
        hist = (kk == GEN) | (kk == CPY) | (kk == KNOW)
        other_ok = ~(q_loc & hist) | (ks == qq)
    else:
        other_ok = np.ones_like(causal)
    after = np.where(kk == LOC, loc_ok, np.where(kk == C, c_ok, other_ok))
    vis = causal & (q_prefix | after) & (kk != PAD)
    pad_q = (qk == PAD)
    return np.where(pad_q, ki == qi, vis)


@dataclass
class Seq:
    ids: np.ndarray
    pos: np.ndarray
    kind: np.ndarray
    rnd: np.ndarray
    seg: np.ndarray
    open_at: np.ndarray
    stp: np.ndarray
    qref: np.ndarray
    target: np.ndarray
    opts: MaskOpts = field(default_factory=MaskOpts)
    view: str = "chain"

    def __len__(self) -> int:
        return len(self.ids)

    def rows(self, q_lo: int, q_hi: int, k_hi: int | None = None) -> np.ndarray:
        k_hi = len(self) if k_hi is None else k_hi
        qs, ks = slice(q_lo, q_hi), slice(0, k_hi)
        return visible(np.arange(q_lo, q_hi), self.kind[qs], self.rnd[qs], self.qref[qs],
                       np.arange(k_hi), self.kind[ks], self.rnd[ks], self.open_at[ks], self.stp[ks], self.opts)

    def mask_dense(self) -> np.ndarray:
        return self.rows(0, len(self))

    def target_index(self) -> np.ndarray:
        """Storage indices t whose token is a training target (predicted from t - 1)."""
        return np.nonzero(self.target)[0]

    def prompt_ids(self) -> list[int]:
        """The sequence a plain engine (vLLM) should see: hidden <locate> originals dropped, copies in their place.

        Valid for views whose masked tokens see everything that is left (masked view; locate / nomask chain views),
        i.e. not for the sealedread chain. Positions of what is left are contiguous by construction."""
        keep = self.kind != LOC
        pos = self.pos[keep]
        if len(pos) and not np.array_equal(pos, np.arange(pos[0], pos[0] + len(pos))):
            raise ValueError("prompt positions are not contiguous (a skipped C?)")
        return self.ids[keep].tolist()


class SeqBuilder:
    """Builds a Seq piece by piece; also used incrementally by the reference engine (append, truncate)."""

    def __init__(self, tok: Tok, opts: MaskOpts | None = None, view: str = "chain"):
        self.tok = tok
        self.opts = opts or MaskOpts()
        self.view = view
        self.ids: list[int] = []
        self.pos: list[int] = []
        self.kind: list[int] = []
        self.rnd: list[int] = []
        self.seg: list[int] = []
        self.open_at: list[int] = []
        self.stp: list[int] = []
        self.qref: list[int] = []
        self.target: list[bool] = []
        self.next_pos = 0
        self.round = 0
        self.step = -1
        self.cur_qref = -1
        self.c_spans: dict[int, list[int]] = {}
        self.c_tokens = 0                 # tokens of C, materialized or skipped
        self.version = 0                  # bumped when earlier metadata changes (daft opens a segment)

    def __len__(self) -> int:
        return len(self.ids)

    def add_ids(self, ids: list[int], kind: int, target: bool = False, seg: int = -1,
                pos: list[int] | None = None) -> tuple[int, int]:
        start = len(self.ids)
        if pos is not None and len(pos) != len(ids):
            raise ValueError(f"{len(ids)} tokens for {len(pos)} positions")
        for j, t in enumerate(ids):
            self.ids.append(int(t))
            if pos is None:
                self.pos.append(self.next_pos)
                self.next_pos += 1
            else:
                self.pos.append(int(pos[j]))
            self.kind.append(kind)
            self.rnd.append(self.round)
            self.seg.append(seg)
            self.open_at.append(NEVER)
            self.stp.append(self.step)
            self.qref.append(self.cur_qref if kind == LOC else -1)
            self.target.append(bool(target))
        if kind == C:
            self.c_tokens += len(ids)
            if seg >= 0:
                self.c_spans.setdefault(seg, []).extend(range(start, len(self.ids)))
        return start, len(self.ids)

    def add(self, text: str, kind: int, target: bool = False, seg: int = -1) -> tuple[int, int]:
        return self.add_ids(self.tok.encode(text), kind, target, seg)

    def add_pieces(self, pieces: list[str], kind: int, target: bool = False,
                   pos: list[int] | None = None) -> tuple[int, int]:
        ids: list[int] = []
        for p in pieces:
            ids.extend(self.tok.encode(p))
        return self.add_ids(ids, kind, target, pos=pos)

    def skip(self, n: int) -> None:
        """Advance the positions without tokens (C not materialized, full frame)."""
        self.next_pos += n
        self.c_tokens += n

    def open(self, sids: list[int], at: int) -> None:
        """daft: the C tokens of these segments become visible from storage index `at` on."""
        for sid in sids:
            for i in self.c_spans.get(sid, []):
                if self.open_at[i] == NEVER:
                    self.open_at[i] = at
        self.version += 1

    def truncate(self, n: int) -> None:
        """Drop tokens from index n on (the reference engine re-encodes a step canonically)."""
        for name in ("ids", "pos", "kind", "rnd", "seg", "open_at", "stp", "qref", "target"):
            del getattr(self, name)[n:]
        self.next_pos = (max(self.pos) + 1) if self.pos else 0

    def seq(self) -> Seq:
        return Seq(ids=np.array(self.ids, np.int64), pos=np.array(self.pos, np.int64),
                   kind=np.array(self.kind, np.int64), rnd=np.array(self.rnd, np.int64),
                   seg=np.array(self.seg, np.int64), open_at=np.array(self.open_at, np.int64),
                   stp=np.array(self.stp, np.int64), qref=np.array(self.qref, np.int64),
                   target=np.array(self.target, bool), opts=self.opts, view=self.view)


# ---------------------------------------------------------------------- whole traces
def add_prompt(b: SeqBuilder, s: Sample, prompts: Prompts, with_c: bool = True, frame: str = "full",
               repeat_q: bool = True, c_count: int | None = None) -> None:
    b.add(prompts.head(), A)
    b.add(prompts.question(s), Q)
    pieces = prompts.context_pieces(s)
    if with_c:
        for text, sid in pieces:
            b.add(text, C, seg=sid)
    elif frame == "full":
        b.skip(c_count if c_count is not None else sum(len(b.tok.encode(t)) for t, _ in pieces))
    elif frame != "compact":
        raise ValueError(f"unknown frame {frame!r}")
    b.add(prompts.question_repeat(s, repeat_q), QP)
    b.add(prompts.assistant_start(), TP)


def tag_pieces(tag: str, text: str, prompts: Prompts) -> list[str]:
    pieces = [f"<{tag}>", text, f"</{tag}>", "\n"]
    if tag == "answer":
        pieces.append(prompts.eos)
    return pieces


def copy_pieces(st: Step) -> list[str]:
    shown = [st.alias.get(i, i) for i in st.ids] if st.alias else list(st.ids)
    return ["<locate>"] + locate_pieces(sorted(shown)) + ["</locate>"]


def add_step(b: SeqBuilder, s: Sample, st: Step, idx: int, prompts: Prompts) -> None:
    view = b.view
    b.step = idx
    if st.tag in MASKED_TAGS:
        if st.tag == "query":
            b.cur_qref = idx
        b.add_pieces(tag_pieces(st.tag, st.text, prompts), GEN, target=view in ("chain", "masked"))
    elif st.tag == "locate":
        b.round += 1
        if view in ("chain", "locate") and st.by == "model":
            start, end = b.add_pieces(["<locate>"] + locate_pieces(st.ids) + ["</locate>"], LOC,
                                      target=view in ("chain", "locate"))
            cs, _ = b.add_pieces(copy_pieces(st), CPY, pos=b.pos[start:end])
        else:
            cs, _ = b.add_pieces(copy_pieces(st), CPY)
        if b.opts.branch == "daft" and view == "chain":
            b.open(st.ids, at=cs)
    elif st.tag == "knowledge":
        if b.opts.branch == "daft" and view == "chain":
            b.add_pieces(["\n", "<knowledge>", "</knowledge>", "\n"], KNOW)       # evidence = original KV of C
        else:
            b.add_pieces(["\n", "<knowledge>", prompts.knowledge_body(s, st.ids, st.alias), "</knowledge>", "\n"],
                         KNOW)
    else:
        raise ValueError(f"unknown tag {st.tag!r}")


def build_seq(s: Sample, steps: list[Step], tok: Tok, prompts: Prompts, view: str = "chain",
              branch: str = "sealedread", frame: str = "full", repeat_q: bool = True,
              opts: MaskOpts | None = None, c_count: int | None = None) -> Seq:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}")
    if opts is None:
        opts = MaskOpts(branch="nomask" if view == "locate" else branch)
    b = SeqBuilder(tok, opts, view)
    add_prompt(b, s, prompts, with_c=view != "masked", frame=frame, repeat_q=repeat_q, c_count=c_count)
    for i, st in enumerate(steps):
        add_step(b, s, st, i, prompts)
    return b.seq()
