"""vLLM in-process engine (GPU only): prefix caching, LoRA adapters, guided decoding for <locate>, prompt log-probs.

Adapted from keychain-dv's rl/engine.py. Differences: no sleep/wake dance (no RL update in the same process), a
regex constraint per request (vLLM's structured outputs; the class name changed across versions, both are tried),
and `prompt_logprobs` for the parity check against the reference engine (check 6).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenOut:
    token_ids: list[int]
    text: str
    finish_reason: str


def _structured(regex: str):
    """SamplingParams kwargs for a regex constraint, whichever API this vLLM has."""
    try:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(regex=regex)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(regex=regex)}


class VLLMEngine:
    def __init__(self, model: str, max_model_len: int = 32768, gpu_memory_utilization: float = 0.85,
                 tensor_parallel_size: int = 1, enable_lora: bool = False, max_lora_rank: int = 64,
                 seed: int = 0, dtype: str = "bfloat16", quantization: str | None = None,
                 enforce_eager: bool = False, **extra):
        from vllm import LLM

        kw = dict(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                  tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=True, seed=seed, dtype=dtype,
                  enforce_eager=enforce_eager, **extra)
        if quantization:
            kw["quantization"] = quantization
        if enable_lora:
            kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
        self.llm = LLM(**kw)
        self.enable_lora = enable_lora
        self.lora = None
        self._next = 1

    def set_adapter(self, path: str | None) -> None:
        """A new LoRA id per load: no stale weights, and a fresh prefix cache (cached KV depends on the weights)."""
        if path is None:
            self.lora = None
            return
        from vllm.lora.request import LoRARequest

        self.lora = LoRARequest(f"adapter-{self._next}", self._next, str(path))
        self._next += 1

    def generate(self, prompts: list[list[int]], max_tokens: int, temperature: float = 0.0, top_p: float = 1.0,
                 stop: list[str] | None = None, seeds: list[int] | None = None,
                 regex: list[str | None] | None = None) -> list[GenOut]:
        from vllm import SamplingParams

        if not prompts:
            return []
        params = []
        for i in range(len(prompts)):
            kw = dict(max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop or None,
                      include_stop_str_in_output=False, skip_special_tokens=False,
                      seed=None if seeds is None else int(seeds[i]))
            if regex is not None and regex[i]:
                kw.update(_structured(regex[i]))
            params.append(SamplingParams(**kw))
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=self.lora,
                                 use_tqdm=False)
        return [GenOut(list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].finish_reason or "")
                for o in outs]

    def prompt_logprobs(self, prompts: list[list[int]]) -> list[list[float | None]]:
        """log p of every prompt token given the previous ones (None for the first)."""
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sp, lora_request=self.lora,
                                 use_tqdm=False)
        res = []
        for p, o in zip(prompts, outs):
            row = []
            for t, d in zip(p, o.prompt_logprobs):
                row.append(None if d is None else float(d[t].logprob))
            res.append(row)
        return res
