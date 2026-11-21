"""LoRA on the linear layers of our Qwen3, saved in the PEFT layout so vLLM (and PEFT) load it as is.

Saved keys: base_model.model.<module path>.lora_A.weight / lora_B.weight, with adapter_config.json.
Initialization as in PEFT: A ~ kaiming uniform, B = 0, so a new adapter starts as the identity.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
PREFIX = "base_model.model."


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0, dtype=torch.float32):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.scale = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Linear(base.in_features, r, bias=False, device=dev, dtype=dtype)
        self.lora_B = nn.Linear(r, base.out_features, bias=False, device=dev, dtype=dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.enabled:
            return out
        d = self.lora_B(self.lora_A(self.drop(x.to(self.lora_A.weight.dtype))))
        return out + (d * self.scale).to(out.dtype)


def apply_lora(model: nn.Module, r: int, alpha: float, dropout: float = 0.0,
               targets=DEFAULT_TARGETS, dtype=torch.float32) -> list[nn.Parameter]:
    """Wrap every target linear layer; freeze everything else. Returns the trainable parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(mod, child_name, LoRALinear(child, r, alpha, dropout, dtype))
    params = [p for n, p in model.named_parameters() if ".lora_" in n]
    for p in params:
        p.requires_grad_(True)
    if not params:
        raise ValueError("no LoRA target found")
    model.lora_config = {"r": r, "lora_alpha": alpha, "lora_dropout": dropout, "target_modules": list(targets)}
    return params


def lora_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_lora_enabled(model: nn.Module, on: bool) -> None:
    for m in lora_layers(model):
        m.enabled = on


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    out = {}
    for n, p in model.named_parameters():
        if ".lora_A." in n or ".lora_B." in n:
            out[PREFIX + n] = p.detach().to(torch.float32).contiguous().cpu()
    return out


def save_lora(model: nn.Module, out_dir: str | Path, base_model: str = "") -> str:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(lora_state(model), str(out / "adapter_model.safetensors"))
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model, "bias": "none",
           "fan_in_fan_out": False, "inference_mode": True, "modules_to_save": None, "use_rslora": False,
           "use_dora": False, "init_lora_weights": True, **model.lora_config}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    return str(out)


def load_lora(model: nn.Module, path: str | Path, dtype=torch.float32) -> None:
    """Apply (if needed) and load a PEFT-layout adapter into the model."""
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text())
    if not lora_layers(model):
        apply_lora(model, cfg["r"], cfg["lora_alpha"], cfg.get("lora_dropout", 0.0),
                   tuple(cfg["target_modules"]), dtype)
    sd = load_file(str(path / "adapter_model.safetensors"))
    own = dict(model.named_parameters())
    n = 0
    for k, v in sd.items():
        name = k[len(PREFIX):] if k.startswith(PREFIX) else k
        name = name.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
        if name not in own:
            raise KeyError(f"adapter key {k} has no parameter")
        with torch.no_grad():
            own[name].copy_(v.to(own[name].dtype))
        n += 1
    if n != len(lora_layers(model)) * 2:
        raise ValueError(f"adapter has {n} tensors for {len(lora_layers(model))} LoRA layers")
