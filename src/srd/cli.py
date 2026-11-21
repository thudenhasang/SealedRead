"""Small helpers shared by the scripts: config with overrides, token counters, sharding, markers."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from srd.config import Config, load_config
from srd.text import approx_tokens


def config_from_argv(argv: list[str] | None = None) -> tuple[Config, list[str]]:
    """`script.py <config.yaml> [key=value ...] [--flags]` -> (config, remaining flags)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        raise SystemExit("usage: script.py <config.yaml> [key=value ...] [--flag ...]")
    path, rest = argv[0], argv[1:]
    overrides = [a for a in rest if "=" in a and not a.startswith("--")]
    flags = [a for a in rest if a not in overrides]
    return load_config(path, overrides), flags


def to_plain(x):
    if isinstance(x, dict):
        return {k: to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_plain(v) for v in x]
    return x


def token_counter(name: str | None):
    """A function text -> token count: the real tokenizer, or ~4 characters per token ("approx")."""
    if not name or name == "approx":
        return approx_tokens
    from srd.tok import load_tok

    tok = load_tok(name)
    return lambda t: len(tok.encode(t))


def shard_args() -> tuple[int, int]:
    """SHARD / NUM_SHARDS from the environment (scripts/cluster/par.sh sets them, one process per GPU)."""
    return int(os.environ.get("SHARD", "0")), int(os.environ.get("NUM_SHARDS", "1"))


def shard_suffix(path: str | Path) -> Path:
    i, n = shard_args()
    p = Path(path)
    return p if n == 1 else p.with_name(f"{p.stem}.shard{i}of{n}{p.suffix}")


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_plain(obj), indent=1, ensure_ascii=False), encoding="utf-8")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
