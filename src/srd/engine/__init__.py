"""Engines: the episode driver and the policies that write the model's tags.

    driver      runs the protocol (state machine, <locate> checks, paste engine) over any policy
    ref         reference engine: our Qwen3, exact masks of every branch, slow, easy to check
    vllm_policy fast engine on stock vLLM: two streams (masked stream without C, compact frame; locate stream with C)
                for SealedRead, one stream for the no-mask control (docs/decisions.md D1)
    retriever   BM25 / dense + reranker, for the RAG baselines (they replace <locate>)
    plain       full-attention baselines (question after C), copy-then-solve, one-pass RAG
"""
