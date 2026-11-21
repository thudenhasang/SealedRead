"""YAML configs with inheritance and command-line overrides.

A config file may name a parent with `extends: base.yaml` (relative to its own folder); keys are merged recursively,
the child winning. Overrides on the command line use dotted keys: `train.epochs=2 eval.n=4`.
String values may use environment variables with a default, `${DATA:-/content/data/srd}/train.jsonl`, so the same
configs serve Colab (scripts/colab/common.sh exports DATA, RUNS) and the GPU machine (scripts/cluster/env.sh).
(Adapted from keychain-dv.)
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict with attribute access (`cfg.train.steps`), recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def _wrap(x: Any) -> Any:
    if isinstance(x, dict):
        return Config({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x


def _merge(base: dict, child: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_raw(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        raw = _merge(_load_raw((path.parent / parent).resolve()), raw)
    return raw


def _parse_value(s: str) -> Any:
    return yaml.safe_load(s)


_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(x: Any) -> Any:
    if isinstance(x, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), x)
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = _load_raw(Path(path).resolve())
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        cur = raw
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = _parse_value(value)
    return _wrap(_expand(raw))
