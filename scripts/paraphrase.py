"""Type-2 samples (table 4): rewrite the question with a strong model, keep it only if an answerer that reads the
gold segments still gives the same answer and every name is kept (GPU, vLLM). Rewrites the files in place (the
original is kept as <file>.orig).

    python scripts/paraphrase.py configs/teacher.yaml /content/data/srd/train.jsonl /content/data/srd/val.jsonl \
        /content/data/srd/reserve.jsonl
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srd.cli import config_from_argv, log, to_plain  # noqa: E402
from srd.data import paraphrase as P  # noqa: E402
from srd.data.sample import load_samples, write_jsonl  # noqa: E402
from srd.engine.factory import make_vllm_engine  # noqa: E402
from srd.text import exact_match  # noqa: E402
from srd.tok import load_tok  # noqa: E402


def chat(t: str, prefix: str) -> str:
    return f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n{prefix}"


def main() -> int:
    cfg, files = config_from_argv()
    ecfg = to_plain(cfg.engine)
    tok = load_tok(ecfg.get("tokenizer") or ecfg["model"])
    eng = make_vllm_engine({**ecfg, "max_model_len": 8192}, enable_lora=False)
    prefix = cfg.prompts.get("assistant_prefix", "")
    for f in files:
        ss = load_samples(f)
        todo = [i for i, s in enumerate(ss) if s.meta.get("needs_paraphrase")]
        for attempt in range(2):
            if not todo:
                break
            rw = eng.generate([tok.encode(chat(P.REWRITE.format(facts=P.gold_facts(ss[i]), question=ss[i].question),
                                               prefix)) for i in todo], max_tokens=96, temperature=0.7,
                              stop=["<|im_end|>"], seeds=[attempt * 100003 + i for i in todo])
            news = [o.text.strip().splitlines()[0] if o.text.strip() else "" for o in rw]
            ans = eng.generate([tok.encode(chat(P.ANSWER.format(facts=P.gold_facts(ss[i]), question=q), prefix))
                                for i, q in zip(todo, news)], max_tokens=32, temperature=0.0, stop=["<|im_end|>"])
            left = []
            for i, q, a in zip(todo, news, ans):
                s = ss[i]
                names = [g.title for g in s.segments if g.gold]
                if q and P.keeps_names(s.question, q, names) and exact_match(a.text.strip(), s.answers):
                    new = P.apply(s, q)
                    new.meta.update(answer_kept=True, needs_paraphrase=False)
                    ss[i] = new
                else:
                    left.append(i)
            todo = left
        for i in todo:                         # no acceptable rewrite: back to a plain multi-hop sample
            ss[i].qtype = "multihop"
            ss[i].meta.update(needs_paraphrase=False, paraphrase_failed=True)
        shutil.copy2(f, f + ".orig")
        write_jsonl(f, [s.to_dict() for s in ss])
        n = sum(1 for s in ss if s.qtype == "paraphrase")
        log(f"{Path(f).name}: {n} paraphrased, {len(todo)} reverted")
    print("PARAPHRASE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
