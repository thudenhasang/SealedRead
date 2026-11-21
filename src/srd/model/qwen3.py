"""A small Qwen3 (dense, full attention) with explicit masks, custom position IDs and a growable KV cache.

Why not transformers: every token here needs its own visibility (eq. 1) and its own position ID (copies reuse the
positions of their original; the masked view skips |C| positions), and the reference engine attends a *subset* of
the cache. This file is short enough to check by hand, loads the official safetensors weights by their names, and
tests/test_model.py checks it against transformers' Qwen3ForCausalLM on random weights.

Architecture: embed -> N x [RMSNorm -> attention (q/k RMSNorm per head, RoPE, GQA) -> residual -> RMSNorm ->
SwiGLU MLP -> residual] -> RMSNorm -> lm_head (tied to the embedding in the 0.6B/1.7B/4B checkpoints).
Hybrid models (Qwen3.5: Gated DeltaNet layers) are excluded by design (condition (i) of property 1).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from srd.model.attention import Attn, attend


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    rope_scaling: dict | None = None
    max_position_embeddings: int = 32768

    def __post_init__(self):
        if not self.head_dim:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @staticmethod
    def from_dict(d: dict) -> "Qwen3Config":
        d = dict(d)
        rp = d.get("rope_parameters") or {}
        if "rope_theta" not in d and "rope_theta" in rp:
            d["rope_theta"] = rp["rope_theta"]
        if not d.get("rope_scaling") and rp.get("rope_type") not in (None, "default"):
            d["rope_scaling"] = rp
        if d.get("layer_types") and any(t != "full_attention" for t in d["layer_types"]):
            raise ValueError("hybrid / sliding-window layers are not supported (condition (i))")
        names = {f.name for f in fields(Qwen3Config)}
        return Qwen3Config(**{k: v for k, v in d.items() if k in names and v is not None})

    @staticmethod
    def tiny(vocab_size: int = 259, layers: int = 2) -> "Qwen3Config":
        return Qwen3Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_theta=10000.0,
                           tie_word_embeddings=True, max_position_embeddings=1 << 20)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def _yarn(cfg: Qwen3Config, dim: int) -> tuple[torch.Tensor, float]:
    """YaRN inverse frequencies and attention factor (as in transformers' _compute_yarn_parameters)."""
    rs = cfg.rope_scaling
    base, factor = cfg.rope_theta, float(rs["factor"])
    orig = rs.get("original_max_position_embeddings") or cfg.max_position_embeddings
    att = rs.get("attention_factor") or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)
    beta_fast, beta_slow = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1

    def corr(n_rot):
        return (dim * math.log(orig / (n_rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr(beta_fast)), 0)
    high = min(math.ceil(corr(beta_slow)), dim - 1)
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extra, inter = 1.0 / pos_freqs, 1.0 / (factor * pos_freqs)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    extra_factor = 1 - ramp
    return inter * (1 - extra_factor) + extra * extra_factor, float(att)


class Rotary(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("inv_freq", torch.empty(cfg.head_dim // 2), persistent=False)
        self.scale = 1.0
        self.reset()

    def reset(self, device=None) -> None:
        cfg, dim = self.cfg, self.cfg.head_dim
        rs = cfg.rope_scaling or {}
        kind = rs.get("rope_type") or rs.get("type") or "default"
        if kind == "yarn":
            inv, self.scale = _yarn(cfg, dim)
        elif kind == "default":
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        else:
            raise ValueError(f"rope type {kind!r} not supported")
        dev = torch.device(device) if device is not None else self.inv_freq.device
        if inv.is_meta and dev.type != "meta":
            return                       # built under torch.device("meta"): from_pretrained calls reset("cpu")
        self.inv_freq = inv.to(dev)

    def forward(self, pos: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = pos.float()[..., None] * self.inv_freq.float()[None, None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return (emb.cos() * self.scale).to(dtype), (emb.sin() * self.scale).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class KVCache:
    """Per-layer K/V in storage order, preallocated and grown by doubling; `truncate` rolls back."""

    def __init__(self, n_layers: int):
        self.k: list[torch.Tensor | None] = [None] * n_layers
        self.v: list[torch.Tensor | None] = [None] * n_layers
        self.length = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.length, self.length + k.shape[2]
        buf = self.k[layer]
        if buf is None or buf.shape[2] < end:
            cap = max(end, 2 * (buf.shape[2] if buf is not None else 0), 256)
            nk = k.new_zeros(k.shape[0], k.shape[1], cap, k.shape[3])
            nv = v.new_zeros(v.shape[0], v.shape[1], cap, v.shape[3])
            if buf is not None and start:
                nk[:, :, :start] = self.k[layer][:, :, :start]
                nv[:, :, :start] = self.v[layer][:, :, :start]
            self.k[layer], self.v[layer] = nk, nv
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]

    def advance(self, n: int) -> None:
        self.length += n

    def truncate(self, n: int) -> None:
        if n > self.length:
            raise ValueError("cannot truncate forward")
        self.length = n

    def clone(self) -> "KVCache":
        out = KVCache(len(self.k))
        out.length = self.length
        out.k = [None if t is None else t[:, :, :self.length].clone() for t in self.k]
        out.v = [None if t is None else t[:, :, :self.length].clone() for t in self.v]
        return out


class Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, idx: int):
        super().__init__()
        self.idx = idx
        self.h, self.hkv, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        b = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.h * self.d, bias=b)
        self.k_proj = nn.Linear(cfg.hidden_size, self.hkv * self.d, bias=b)
        self.v_proj = nn.Linear(cfg.hidden_size, self.hkv * self.d, bias=b)
        self.o_proj = nn.Linear(self.h * self.d, cfg.hidden_size, bias=b)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, spec: Attn, cache: KVCache | None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.h, self.d)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.hkv, self.d)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.hkv, self.d).transpose(1, 2)
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        if cache is not None:
            k, v = cache.update(self.idx, k, v)
        out = attend(q, k, v, spec)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.h * self.d))


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, cfg: Qwen3Config, idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg, idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, spec, cache):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, spec, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class Body(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = Rotary(cfg)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.model = Body(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.grad_ckpt = False

    @property
    def device(self):
        return self.model.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.num_hidden_layers)

    def forward(self, ids: torch.Tensor, pos: torch.Tensor, attn: Attn | None = None,
                cache: KVCache | None = None) -> torch.Tensor:
        """ids, pos [B, T] -> final hidden states [B, T, H] (call `logits` on the rows you need)."""
        attn = attn or Attn("causal")
        x = self.model.embed_tokens(ids)
        cos, sin = self.model.rotary(pos, x.dtype)
        use_ckpt = self.grad_ckpt and self.training and cache is None and torch.is_grad_enabled()
        for layer in self.model.layers:
            if use_ckpt:
                x = checkpoint(layer, x, cos, sin, attn, None, use_reentrant=False)
            else:
                x = layer(x, cos, sin, attn, cache)
        if cache is not None:
            cache.advance(ids.shape[1])
        return self.model.norm(x)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


# ---------------------------------------------------------------------- weights
WEIGHT_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab.json", "tokenizer*"]


def resolve(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name_or_path, allow_patterns=WEIGHT_PATTERNS))


def load_state(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    sd: dict[str, torch.Tensor] = {}
    for f in sorted(path.glob("*.safetensors")):
        sd.update(load_file(str(f)))
    if not sd:
        raise FileNotFoundError(f"no safetensors under {path}")
    return sd


def from_pretrained(name_or_path: str, dtype=torch.bfloat16, device="cpu") -> Qwen3ForCausalLM:
    path = resolve(name_or_path)
    cfg = Qwen3Config.from_dict(json.loads((path / "config.json").read_text()))
    with torch.device("meta"):
        model = Qwen3ForCausalLM(cfg)
    sd = {k: v.to(dtype) for k, v in load_state(path).items()}
    if cfg.tie_word_embeddings:
        sd.pop("lm_head.weight", None)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if not (cfg.tie_word_embeddings and m == "lm_head.weight")]
    if missing or unexpected:
        raise ValueError(f"weights do not match: missing {missing[:5]}, unexpected {unexpected[:5]}")
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.rotary.reset("cpu")
    model.source = str(path)
    return model.to(device)


def save_pretrained(model: Qwen3ForCausalLM, out_dir: str | Path, source_dir: str | Path | None = None,
                    shard_bytes: int = 4 << 30) -> None:
    """Weights in the Hugging Face layout (vLLM can serve the folder); tokenizer/config copied from the source."""
    import shutil

    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    if model.cfg.tie_word_embeddings:
        sd.pop("lm_head.weight", None)
    shards, cur, size = [], {}, 0
    for k, v in sd.items():
        nb = v.numel() * v.element_size()
        if cur and size + nb > shard_bytes:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = v
        size += nb
    shards.append(cur)
    index = {}
    for i, part in enumerate(shards):
        fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(part, str(out / fn), metadata={"format": "pt"})
        index.update({k: fn for k in part})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    src = Path(source_dir) if source_dir else Path(getattr(model, "source", ""))
    if src and src.is_dir():
        for f in src.iterdir():
            if f.suffix in (".json", ".txt", ".model") and not f.name.startswith("model.safetensors"):
                shutil.copy2(f, out / f.name)
