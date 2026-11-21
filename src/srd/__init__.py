"""SealedRead: the reasoning phase never sees the long context C; only <locate> does, and it can only emit IDs.

Layout of the package (docs/README.md has the map from the proposal's sections to the code):
    protocol    tags, steps, state machine, validity rules, paste engine (text level)
    grammar     the <locate> grammar (token trie for the reference engine, regex for vLLM)
    prompts     instruction A, question/context rendering, chat pieces
    layout      token layout of a trace and the attention rule of every branch (one source of truth)
    reward      R_mu (eq. 4/5), F2, Acc, filters
    data/       shards -> chunked, ID-labelled, distractor-padded samples; fictional substitution; data checks
    model/      a small Qwen3 implementation with explicit masks, positions and KV cache; LoRA
    engine/     episode driver, reference engine, vLLM two-stream engine, retrievers, plain baselines
    train/      SFT on traces (FlexAttention masks), checkpoints
    eval/       metrics, paired cluster bootstrap, pre-registered gates, mask probes
"""
__version__ = "0.1.0"
