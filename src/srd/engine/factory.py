"""Build a policy from a config section (scripts share this).

engine:
  backend: vllm | ref         vllm: stock vLLM (two-stream / one stream); ref: reference engine (any branch, slow)
  model: Qwen/Qwen3-4B-Instruct-2507
  adapter: runs/<run>/best/adapter    (optional LoRA in PEFT layout; or a full model folder as `model`)
  max_model_len, gpu_memory_utilization, tensor_parallel_size, quantization...
  temperature, top_p
Branch -> how the episode runs:
  sealedread        vllm two_stream | ref chain (sealedread mask)        locator: model
  sealedread_chain  ref chain (the main design, full frame)               locator: model
  nomask            vllm nomask | ref chain (nomask)                      locator: model
  daft              ref chain (daft)                                      locator: model
  rag               vllm masked | ref masked (compact)                    locator: retriever
"""
from __future__ import annotations

from srd.engine.driver import DriverConfig
from srd.prompts import Prompts
from srd.protocol import ProtoConfig
from srd.tok import load_tok

VLLM_MODE = {"sealedread": "two_stream", "nomask": "nomask", "rag": "masked"}


def make_vllm_engine(ecfg: dict, enable_lora: bool):
    from srd.engine.vllm_engine import VLLMEngine

    keys = ("max_model_len", "gpu_memory_utilization", "tensor_parallel_size", "quantization", "max_lora_rank",
            "seed", "dtype", "enforce_eager")
    return VLLMEngine(ecfg["model"], enable_lora=enable_lora, **{k: ecfg[k] for k in keys if k in ecfg})


def load_ref_model(ecfg: dict, device=None):
    import torch

    from srd.model.lora import load_lora
    from srd.model.qwen3 import from_pretrained

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if ecfg["model"] == "tiny":                       # CPU tests of the scripts: a seeded random tiny model
        from srd.model.qwen3 import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config.tiny()).to(device)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.2)
        return model.eval()
    model = from_pretrained(ecfg["model"], dtype=getattr(torch, ecfg.get("dtype", "bfloat16")), device=device)
    if ecfg.get("adapter"):
        load_lora(model, ecfg["adapter"])
        model.to(device)
    return model.eval()


def make_policy(ecfg: dict, branch: str, proto: ProtoConfig, prompts: Prompts | None = None, engine=None,
                ref_model=None, opts=None, repeat_q: bool = True, frame: str = "full", locate_query_only: bool = False):
    """Returns (policy, tokenizer, locator)."""
    prompts = prompts or Prompts()
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    backend = ecfg.get("backend", "vllm")
    t, p = ecfg.get("temperature", 0.0), ecfg.get("top_p", 1.0)
    locator = "retriever" if branch == "rag" else "model"
    if backend == "vllm":
        from srd.engine.vllm_policy import VLLMPolicy

        if branch not in VLLM_MODE:
            raise ValueError(f"branch {branch!r} needs the reference engine (backend: ref)")
        if engine is None:
            engine = make_vllm_engine(ecfg, enable_lora=bool(ecfg.get("adapter")))
            engine.set_adapter(ecfg.get("adapter"))
        pol = VLLMPolicy(engine, tok, prompts, VLLM_MODE[branch], proto, t, p, repeat_q, locate_query_only)
        return pol, tok, locator
    if backend == "ref":
        from srd.engine.ref import RefPolicy
        from srd.layout import MaskOpts

        model = ref_model or load_ref_model(ecfg)
        mask_branch = {"sealedread": "sealedread", "sealedread_chain": "sealedread", "nomask": "nomask",
                       "daft": "daft", "rag": "sealedread"}[branch]
        view = "masked" if branch == "rag" else "chain"
        pol = RefPolicy(model, tok, prompts, branch=mask_branch, view=view,
                        frame="compact" if branch == "rag" else frame, repeat_q=repeat_q,
                        opts=opts or MaskOpts(branch=mask_branch), proto=proto, temperature=t, top_p=p)
        return pol, tok, locator
    raise ValueError(f"unknown backend {backend!r}")


def driver_config(proto: ProtoConfig, locator: str, lenient: bool = False, permute_ids: bool = False) -> DriverConfig:
    return DriverConfig(proto=proto, locator=locator, lenient=lenient, permute_ids=permute_ids)
