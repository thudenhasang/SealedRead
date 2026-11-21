"""Get the raw multi-hop QA shards (HotpotQA, 2WikiMultihopQA, MuSiQue) that scripts/build_data.py reads.

Canonical source: ragproj2's shards on Google Drive (MyDrive/ragproj2/data/train), built from the official training
splits (s0-s4: 1,000 questions each; MuSiQue s10-s14: 3-4 hop questions) and dev splits without HippoRAG 2's test
questions (s90). On Colab with Drive mounted:

    python scripts/fetch_data.py --from-drive /content/drive/MyDrive/ragproj2/data/train --out /content/data/shards

Without that Drive folder, rebuild equivalent shards from the Hugging Face copies of the official splits (same
pinned revisions as ragproj2; s90 then does not exclude HippoRAG 2's test questions):

    python scripts/fetch_data.py --from-hf --out /content/data/shards [--shards 5]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

SOURCES = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "1908d6afbbead072334abe2965f91bd2709910ab",
                 ["distractor/train-00000-of-00002.parquet", "distractor/train-00001-of-00002.parquet"],
                 ["distractor/validation-00000-of-00001.parquet"]),
    "2wikimultihopqa": ("framolfese/2WikiMultihopQA", "fe713bfbd1afbca1a65246741a75890405d56a3a",
                        ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
                        ["data/validation-00000-of-00001.parquet"]),
    "musique": ("bdsaglam/musique", "22873a405dd809893b22ada0b499299fb612d2df",
                ["musique_ans_v1.0_train.jsonl"], ["musique_ans_v1.0_dev.jsonl"]),
}


def from_drive(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*_s*.json")):
        if p.name.endswith("_corpus.json"):
            continue                    # the pooled corpora are not needed: contexts are built per sample
        dst = out / p.name
        if not dst.exists() or dst.stat().st_size != p.stat().st_size:
            shutil.copy2(p, dst)
        n += 1
    print(f"{n} shard files in {out}")


def _records(name: str, files: list[str], repo: str, rev: str) -> list[dict]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    paths = [hf_hub_download(repo, f, repo_type="dataset", revision=rev) for f in files]
    if name == "musique":
        return [json.loads(line) for p in paths for line in open(p, encoding="utf-8")]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out = []
    for r in df.itertuples(index=False):
        s = {"_id": r.id, "question": r.question, "answer": r.answer, "type": r.type,
             "supporting_facts": [[t, int(i)] for t, i in zip(r.supporting_facts["title"], r.supporting_facts["sent_id"])],
             "context": [[t, list(sents)] for t, sents in zip(r.context["title"], r.context["sentences"])]}
        if name == "hotpotqa":
            s["level"] = r.level
        else:
            s["evidences"] = [list(e) for e in r.evidences]
        out.append(s)
    return out


def from_hf(out: Path, shards: int, per_shard: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, (repo, rev, train_files, dev_files) in SOURCES.items():
        for split, files, ks in (("train", train_files, range(shards)), ("dev", dev_files, [90])):
            recs = _records(name, files, repo, rev)
            order = list(range(len(recs)))
            random.Random(seed).shuffle(order)
            for j, k in enumerate(ks):
                part = [recs[i] for i in order[j * per_shard:(j + 1) * per_shard]]
                (out / f"{name}_s{k}.json").write_text(json.dumps(part, ensure_ascii=False), encoding="utf-8")
                print(f"{name}_s{k}: {len(part)} questions ({split})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-drive", default=None)
    ap.add_argument("--from-hf", action="store_true")
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--per-shard", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.from_drive:
        from_drive(Path(args.from_drive), Path(args.out))
    elif args.from_hf:
        from_hf(Path(args.out), args.shards, args.per_shard, args.seed)
    else:
        ap.error("give --from-drive <folder> or --from-hf")
    print("FETCH_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
