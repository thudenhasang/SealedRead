"""G0: tokenizer assumptions the layout relies on (CPU; downloads the tokenizer only).

    python scripts/tokenizer_check.py Qwen/Qwen3-4B-Instruct-2507 [--out runs/tokenizer_check.json]
Checks: tags are not special tokens and round-trip; '[[S_k]]' labels are prefix-free for k = 1..9999 (the <locate>
trie); labels with the same number of digits have the same length (ID permutation probe); a sorted copy has the
length of its original (copies reuse positions); a 3-ID <locate> block fits the safety cap; the chat markers are
single tokens. Also reports how often canonical piece tokenization differs from tokenizing the joined text.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import write_json  # noqa: E402
from srd.grammar import LocateTrie  # noqa: E402
from srd.protocol import TAGS, label, locate_pieces  # noqa: E402
from srd.tok import encode_pieces, load_tok  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tok = load_tok(args.tokenizer)
    rep, ok = {}, True
    special = set(getattr(getattr(tok, "tk", None), "all_special_ids", []) or [])
    for t in TAGS:
        for s in (f"<{t}>", f"</{t}>"):
            ids = tok.encode(s)
            bad = bool(set(ids) & special) or tok.decode(ids) != s
            rep[s] = {"ids": ids, "bad": bad}
            ok &= not bad
    LocateTrie(tok, list(range(1, 10000)), 3)            # raises if labels are not prefix-free
    rep["labels_prefix_free"] = True
    by_digits: dict[int, Counter] = {}
    for k in range(1, 10000):
        by_digits.setdefault(len(str(k)), Counter())[len(tok.encode(label(k)))] += 1
    rep["label_lengths_by_digits"] = {d: dict(c) for d, c in by_digits.items()}
    same = all(len(c) == 1 for c in by_digits.values())
    rep["same_length_per_digit_count"] = same
    ok &= same
    rng = random.Random(0)
    worst, mism = 0, 0
    for _ in range(5000):
        ids = rng.sample(range(1, 10000), rng.randint(1, 3))
        a = encode_pieces(tok, ["<locate>"] + locate_pieces(ids) + ["</locate>"])
        b = encode_pieces(tok, ["<locate>"] + locate_pieces(sorted(ids)) + ["</locate>"])
        mism += len(a) != len(b)
        worst = max(worst, len(a))
    rep["copy_length_mismatches"] = mism
    rep["max_locate_tokens"] = worst
    ok &= mism == 0 and worst <= 48
    for m in ("<|im_start|>", "<|im_end|>"):
        rep[m] = tok.encode(m)
        ok &= len(rep[m]) == 1
    texts = ["Need (1) the director. alpha=query", "S_871: the director is Mara Elling. beta=valid_to_think",
             "Check: S_871 -> Mara Elling; S_5126 -> Porto. alpha=answer", "Porto"]
    diff = sum(tok.encode(f"<reason>{t}</reason>\n") != encode_pieces(tok, ["<reason>", t, "</reason>", "\n"])
               for t in texts)
    rep["canonical_vs_joined_differences"] = f"{diff}/{len(texts)} (informational: canonical pieces are used everywhere)"
    rep["pass"] = ok
    for k, v in rep.items():
        print(f"{k}: {v}")
    if args.out:
        write_json(args.out, rep)
    print("TOKENIZER_CHECK_PASS" if ok else "TOKENIZER_CHECK_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
