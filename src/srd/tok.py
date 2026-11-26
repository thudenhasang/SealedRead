"""Tokenizers behind one small interface.

Every token sequence in the project is built from *pieces* (a tag, a free-text span, a newline...) that are tokenized
separately and concatenated ("canonical tokenization"). Training, the reference engine and the vLLM engine all use
it, so a trace has exactly one token sequence wherever it is used, and a <locate> copy always has as many tokens as
the original (docs/03-protocol.md).

    HFTok    a Hugging Face tokenizer (the real model)
    ByteTok  UTF-8 bytes + the chat special tokens, for CPU tests with tiny random models
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class Tok(Protocol):
    name: str
    vocab_size: int
    eos_text: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def eos_ids(self) -> set[int]: ...


class HFTok:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.name = name
        self.tk = AutoTokenizer.from_pretrained(name)
        self.vocab_size = len(self.tk)
        self.eos_text = "<|im_end|>"
        self._cache = lru_cache(maxsize=200_000)(self._encode)

    def _encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tk.encode(text, add_special_tokens=False))

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if len(text) > 2000:                      # long pieces (context segments) are not worth caching
            return list(self._encode(text))
        return list(self._cache(text))

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)

    def eos_ids(self) -> set[int]:
        out = {self.tk.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        return {i for i in out if isinstance(i, int) and i >= 0}


class ByteTok:
    """Bytes 0..255, then the special tokens. Deterministic, no download: the tokenizer of the CPU tests."""

    SPECIAL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")

    def __init__(self):
        self.name = "bytes"
        self.vocab_size = 256 + len(self.SPECIAL)
        self.eos_text = "<|im_end|>"

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for j, sp in enumerate(self.SPECIAL):
                if text.startswith(sp, i):
                    out.append(256 + j)
                    i += len(sp)
                    break
            else:
                nxt = min((p for p in (text.find(sp, i) for sp in self.SPECIAL) if p >= 0), default=len(text))
                out.extend(text[i:nxt].encode("utf-8"))
                i = nxt
        return out

    def decode(self, ids: list[int]) -> str:
        out, buf = [], bytearray()
        for t in ids:
            if t < 256:
                buf.append(t)
            else:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = bytearray()
                out.append(self.SPECIAL[t - 256])
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def eos_ids(self) -> set[int]:
        return {257, 258}


def load_tok(name: str) -> Tok:
    return ByteTok() if name in ("bytes", "byte", "tiny") else HFTok(name)


def encode_pieces(tok: Tok, pieces: list[str]) -> list[int]:
    out: list[int] = []
    for p in pieces:
        out.extend(tok.encode(p))
    return out
