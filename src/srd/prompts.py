"""Instruction A, the question and context blocks, and the chat pieces around them (section 3.1).

Input layout (sandwich): [A] [Q] [C] [Q'] [T']
    A   system turn + start of the user turn + the protocol instructions (kept first: attention sink)
    Q   the question, before C (C is encoded having seen it, which helps <locate>)
    C   the segments, each '[[S_k]] (Title / Section) text [[/S_k]]'
    Q'  the question again, after C, encoded with C masked (brings it next to the reasoning), closes the user turn
    T'  the template that opens the assistant turn
"""
from __future__ import annotations

from srd.data.sample import Sample, Segment
from srd.protocol import close_label, label

SYSTEM = "You are a careful assistant that answers questions about long documents."

INSTRUCTION = """\
You answer a question about a long set of documents. The documents are split into segments labelled [[S_k]] ... \
[[/S_k]], where k is an ID. While you think you cannot see the documents: you can only read the segments you ask for.

Write your solution as a sequence of tags. What goes in each tag:
- reason: in your own words, what you know so far and what is still missing. Its last words are exactly \
alpha=query (you need to look something up) or alpha=answer (you have enough evidence).
- query: the one thing to look up now.
- locate: 1 to 3 segment IDs, written as labels separated by spaces.
- knowledge: written for you, not by you: the segments you located, pasted word for word.
- recheck: check the pasted text against your query. Its last words are exactly beta=valid_to_think (it answers \
the query), beta=explore_query (it helps partly) or beta=reject_and_requery (it does not help: ask again).
- answer: the final answer only, as short as possible.

Example (another question; its documents are not shown here):
<reason>The question asks where the director of Film X was born. I need the director first, then the birthplace. \
alpha=query</reason>
<query>director of Film X</query>
<locate>[[S_12]]</locate>

<knowledge>
[[S_12]] (Film X / Production) Film X is a 1998 drama directed by Ana Ruiz. [[/S_12]]
</knowledge>
<recheck>S_12 names the director of Film X: Ana Ruiz. beta=valid_to_think</recheck>
<reason>The director is Ana Ruiz. I still need where she was born. alpha=query</reason>
<query>birthplace of Ana Ruiz</query>
<locate>[[S_407]] [[S_88]]</locate>

<knowledge>
[[S_88]] (Ana Ruiz / Early life) Ana Ruiz was born in Coimbra and studied in Lisbon. [[/S_88]]
[[S_407]] (Ana Ruiz / Career) Ruiz moved to Madrid in 1990. [[/S_407]]
</knowledge>
<recheck>S_88 says Ana Ruiz was born in Coimbra; S_407 is not needed. beta=valid_to_think</recheck>
<reason>Check: Film X was directed by Ana Ruiz (S_12), and Ana Ruiz was born in Coimbra (S_88). alpha=answer</reason>
<answer>Coimbra</answer>

Rules: write your own content, never these descriptions or the example; never ask again for a segment that is \
already pasted; before answering, check every step of your reasoning against the pasted text in a final <reason>; \
if the question needs no document (e.g. arithmetic), answer directly."""

PLAIN_INSTRUCTION = """\
Answer the question about the documents below. Think step by step in <reason> ... </reason>, then give the final \
answer only, as short as possible, in <answer> ... </answer>."""

COPY_THEN_SOLVE_INSTRUCTION = """\
Answer the question about the documents below. First copy, word for word, the sentences that are needed to answer \
it inside <quote> ... </quote>. Then think step by step in <reason> ... </reason>, and give the final answer only, \
as short as possible, in <answer> ... </answer>."""


class Prompts:
    def __init__(self, assistant_prefix: str = "", show_titles: bool = True, system: str = SYSTEM,
                 instruction: str = INSTRUCTION):
        self.assistant_prefix = assistant_prefix      # e.g. "<think>\n\n</think>\n\n" for a thinking teacher
        self.show_titles = show_titles
        self.system = system
        self.instruction = instruction
        self.eos = "<|im_end|>"

    @staticmethod
    def from_cfg(d: dict | None) -> "Prompts":
        d = dict(d or {})
        return Prompts(assistant_prefix=d.get("assistant_prefix") or "",
                       show_titles=d.get("show_titles", True) is not False)

    # ------------------------------------------------------------------ the sandwich
    def head(self) -> str:
        return (f"<|im_start|>system\n{self.system}<|im_end|>\n"
                f"<|im_start|>user\n{self.instruction}\n\n")

    def question(self, s: Sample) -> str:
        return f"Question: {s.question}\n\n"

    def segment(self, seg: Segment, shown: int | None = None) -> str:
        k = seg.sid if shown is None else shown
        if self.show_titles:
            return f"{label(k)} ({seg.title} / {seg.section}) {seg.text} {close_label(k)}"
        return f"{label(k)} {seg.text} {close_label(k)}"

    def context_pieces(self, s: Sample) -> list[tuple[str, int]]:
        """C as (text, sid) pieces; separators carry sid -1 (they belong to no segment)."""
        out: list[tuple[str, int]] = [("Documents:\n\n", -1)]
        for i, seg in enumerate(s.segments):
            if i:
                out.append(("\n\n", -1))
            out.append((self.segment(seg), seg.sid))
        out.append(("\n\n", -1))
        return out

    def question_repeat(self, s: Sample, repeat: bool = True) -> str:
        return (f"Question (repeated): {s.question}" if repeat else "") + "<|im_end|>\n"

    def assistant_start(self) -> str:
        return "<|im_start|>assistant\n" + self.assistant_prefix

    def knowledge_body(self, s: Sample, ids: list[int], alias: dict[int, int] | None = None) -> str:
        segs = [self.segment(s.by_sid(i), (alias or {}).get(i)) for i in ids]
        return "\n" + "\n".join(segs) + "\n"

    # ------------------------------------------------------------------ plain prompts (full-attention baselines)
    def plain(self, s: Sample, style: str = "plain") -> str:
        """Question after C, the strongest full-attention format (config (a) of S0, SEALONG, copy-then-solve)."""
        instr = COPY_THEN_SOLVE_INSTRUCTION if style == "copy" else PLAIN_INSTRUCTION
        ctx = "".join(t for t, _ in self.context_pieces(s))
        return (f"<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n{instr}\n\n{ctx}"
                f"Question: {s.question}<|im_end|>\n" + self.assistant_start())

    def rag_one_pass(self, s: Sample, segs: list[Segment]) -> str:
        ctx = "\n".join(self.segment(g) for g in segs)
        return (f"<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n{PLAIN_INSTRUCTION}\n\n"
                f"Documents:\n\n{ctx}\n\nQuestion: {s.question}<|im_end|>\n" + self.assistant_start())
