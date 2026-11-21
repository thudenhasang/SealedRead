"""The Qwen3 implementation against transformers, the cache against the full forward, flex against dense, LoRA."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import trace_two_rounds  # noqa: E402
from srd.layout import C, CPY, GEN, KNOW, MaskOpts, build_seq  # noqa: E402
from srd.model.attention import Attn, make_mask_mod, seq_tensors  # noqa: E402
from srd.model.lora import apply_lora, load_lora, save_lora  # noqa: E402
from srd.model.qwen3 import Qwen3Config, Qwen3ForCausalLM  # noqa: E402
from srd.model.run import forward_seq, target_logprobs  # noqa: E402


def tiny(seed=0, vocab=259):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(Qwen3Config.tiny(vocab)).eval()
    for p in m.parameters():                     # random but not degenerate (norm weights not all 1)
        torch.nn.init.normal_(p, std=0.2)
    return m


def test_matches_transformers():
    tr = pytest.importorskip("transformers")
    cfg = Qwen3Config.tiny(300, layers=3)
    hf_cfg = tr.Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, rope_theta=10000.0,
                            tie_word_embeddings=True, max_position_embeddings=4096, attn_implementation="eager")
    torch.manual_seed(1)
    hf = tr.Qwen3ForCausalLM(hf_cfg).eval()
    for p in hf.parameters():
        torch.nn.init.normal_(p, std=0.2)
    ours = Qwen3ForCausalLM(cfg).eval()
    missing, unexpected = ours.load_state_dict(hf.state_dict(), strict=False)
    assert not unexpected and set(missing) <= {"lm_head.weight"}
    ours.lm_head.weight = ours.model.embed_tokens.weight
    ids = torch.randint(0, 300, (2, 37))
    pos = torch.arange(37)[None].expand(2, -1)
    with torch.no_grad():
        a = hf(input_ids=ids).logits
        b = ours.logits(ours(ids, pos))
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()


def test_cache_matches_full_forward(sample, tok, prompts):
    """Incremental decoding with dense rows over the cache == one forward over the whole chain."""
    m = tiny()
    seq = build_seq(sample, trace_two_rounds(sample), tok, prompts, view="chain")
    h_full, t = forward_seq(m, seq, "dense")
    cache = m.new_cache()
    n = len(seq)
    bounds = [0, int((seq.kind <= C).sum())] + list(range(int((seq.kind <= C).sum()) + 50, n, 50)) + [n]
    outs = []
    with torch.no_grad():
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if lo == hi:
                continue
            rows = seq.rows(lo, hi, hi)
            ids = t["ids"][lo:hi][None]
            pos = t["pos"][lo:hi][None]
            if lo == 0:
                spec = Attn("causal")
            else:
                keep = np.nonzero(rows.any(axis=0))[0]
                spec = Attn("dense", mask=torch.as_tensor(rows[:, keep]), kv_index=torch.as_tensor(keep))
            outs.append(m(ids, pos, spec, cache)[0])
    h_inc = torch.cat(outs)
    assert torch.allclose(h_full, h_inc, atol=1e-4), (h_full - h_inc).abs().max()


@pytest.mark.parametrize("opts", [MaskOpts("sealedread"), MaskOpts("nomask"), MaskOpts("daft"),
                                  MaskOpts(hide_locate=False), MaskOpts(locate_history="query_only")])
def test_flex_rule_equals_dense_rule(sample, tok, prompts, opts):
    from torch.nn.attention.flex_attention import create_mask

    seq = build_seq(sample, trace_two_rounds(sample), tok, prompts, view="chain", opts=opts)
    t = seq_tensors(seq, "cpu", pad_to=128)
    n = t["kind"].shape[0]
    dense = create_mask(make_mask_mod(t, opts), None, None, n, n, device="cpu")[0, 0].numpy()
    want = seq.mask_dense()
    assert np.array_equal(dense[:len(seq), :len(seq)], want)
    assert dense[len(seq):].sum() == n - len(seq)          # pad rows: diagonal only


def test_flex_forward_equals_dense(sample, tok, prompts):
    m = tiny()
    seq = build_seq(sample, trace_two_rounds(sample), tok, prompts, view="chain")
    with torch.no_grad():
        hd, _ = forward_seq(m, seq, "dense", pad_to=128)
        hf, _ = forward_seq(m, seq, "flex", pad_to=128)
    assert torch.allclose(hd[:len(seq)], hf[:len(seq)], atol=1e-4)


def _perturb_c(seq, keep_sids, rng):
    """New C content outside the kept segments, same token count (the token-level invariance probe)."""
    ids = seq.ids.copy()
    sel = (seq.kind == C) & ~np.isin(seq.seg, list(keep_sids))
    ids[sel] = rng.integers(32, 127, sel.sum())
    out = type(seq)(**{**seq.__dict__, "ids": ids})
    return out


def test_invariance_sealedread_and_leak_in_nomask(sample, tok, prompts):
    """Property 1 as an implementation test: with SealedRead, the masked tokens do not change when C changes outside
    the pasted segments; with the no-mask control they do; so do they with daft, whose opened segments carry KV
    contextualized by the segments before them (leak channel 2 of table 2, measured by Delta_ctx)."""
    m = tiny()
    tr = trace_two_rounds(sample)
    pasted = set(tr[2].ids) | set(tr[7].ids)
    rng = np.random.default_rng(0)
    for branch, leaks in (("sealedread", False), ("nomask", True), ("daft", True)):
        seq = build_seq(sample, tr, tok, prompts, view="chain", branch=branch)
        alt = _perturb_c(seq, pasted, rng)
        gen = np.isin(seq.kind[seq.target_index()], [GEN])
        a = target_logprobs(m, seq)[torch.as_tensor(gen)]
        b = target_logprobs(m, alt)[torch.as_tensor(gen)]
        diff = (a - b).abs().max().item()
        assert (diff > 1e-3) == leaks, (branch, diff)
    # daft: segments after the last opened one cannot reach the masked tokens (causal prefill + mask)
    seq = build_seq(sample, tr, tok, prompts, view="chain", branch="daft")
    order = sample.order()
    last = max(order[s] for s in pasted)
    end = np.nonzero((seq.kind == C) & (seq.seg == sample.segments[last].sid))[0].max()
    sel = (seq.kind == C) & (np.arange(len(seq)) > end)
    alt = type(seq)(**{**seq.__dict__, "ids": np.where(sel, rng.integers(32, 127, len(seq)), seq.ids)})
    gen = torch.as_tensor(np.isin(seq.kind[seq.target_index()], [GEN]))
    assert (target_logprobs(m, seq)[gen] - target_logprobs(m, alt)[gen]).abs().max() < 1e-4


def test_masked_view_equals_chain_for_masked_tokens(sample, tok, prompts):
    """The masked view (no C, positions skipped) gives the masked tokens exactly what the chain gives them."""
    m = tiny()
    tr = trace_two_rounds(sample)
    chain = build_seq(sample, tr, tok, prompts, view="chain")
    masked = build_seq(sample, tr, tok, prompts, view="masked", frame="full")
    a = target_logprobs(m, chain)[torch.as_tensor(chain.kind[chain.target_index()] == GEN)]
    b = target_logprobs(m, masked)
    assert torch.allclose(a, b, atol=1e-4)
    assert (masked.kind == CPY).any() and (masked.kind == KNOW).any()


def test_lora_identity_and_roundtrip(tmp_path):
    m = tiny()
    ids = torch.randint(0, 259, (1, 20))
    pos = torch.arange(20)[None]
    with torch.no_grad():
        before = m.logits(m(ids, pos))
    params = apply_lora(m, r=4, alpha=8)
    with torch.no_grad():
        assert torch.allclose(before, m.logits(m(ids, pos)), atol=1e-5)
        for p in params:
            p.normal_(std=0.1)
        after = m.logits(m(ids, pos))
    assert not torch.allclose(before, after)
    save_lora(m, tmp_path / "ad", base_model="tiny")
    m2 = tiny()
    load_lora(m2, tmp_path / "ad")
    with torch.no_grad():
        assert torch.allclose(after, m2.logits(m2(ids, pos)), atol=1e-5)
    from safetensors.torch import load_file

    keys = list(load_file(str(tmp_path / "ad" / "adapter_model.safetensors")))
    assert all(k.startswith("base_model.model.model.layers.") for k in keys)
    assert any(k.endswith("self_attn.q_proj.lora_A.weight") for k in keys)


def test_save_and_load_pretrained(tmp_path):
    import json

    from srd.model.qwen3 import from_pretrained, save_pretrained

    m = tiny()
    save_pretrained(m, tmp_path / "w")
    cfg = m.cfg.__dict__.copy()
    (tmp_path / "w" / "config.json").write_text(json.dumps(cfg))
    m2 = from_pretrained(str(tmp_path / "w"), dtype=torch.float32)
    ids = torch.randint(0, 259, (1, 9))
    pos = torch.arange(9)[None]
    with torch.no_grad():
        assert torch.allclose(m.logits(m(ids, pos)), m2.logits(m2(ids, pos)), atol=1e-6)
