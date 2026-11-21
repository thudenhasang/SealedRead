#!/usr/bin/env bash
# The plan of table 7 on the GPU machine, stage by stage. Each stage is one function; run them one at a time and
# read the decision files before going on (the gates are pre-registered: configs/prereg.yaml).
#   bash scripts/cluster/pipeline.sh s0 | s1a | s1b | s2 | s3 | s4 | s5
source "$(dirname "$0")/env.sh"
SEEDS=${SEEDS:-"0 1 2"}
D="data.samples=[$DATA/train.jsonl,$DATA/val.jsonl,$DATA/reserve.jsonl]"
TR="teacher.samples=$DATA/train.jsonl teacher.reserve=$DATA/reserve.jsonl teacher.out=$RUNS/teacher"

s0() {   # measurements without training (section 5.2)
  bash scripts/cluster/par.sh closed_book scripts/closed_book.py configs/base.yaml --samples $DATA/test_64k.jsonl \
    $DATA/test_fict_64k.jsonl --out $RUNS/closed_book/heldout.jsonl
  NUM_SHARDS=4 bash scripts/cluster/par.sh p0_vllm scripts/p0_probe.py configs/s0.yaml --configs a,d,f
  NUM_SHARDS=4 bash scripts/cluster/par.sh p0_ref scripts/p0_probe.py configs/s0.yaml --configs b,c,e
  $PY scripts/report.py p0 $RUNS/s0 --out $RUNS/s0/p0_decision.json
}

s1a() {  # teacher traces (tensor parallel 2: one process per pair of GPUs is simplest; here one process, TP from config)
  $PY scripts/teacher_traces.py configs/teacher.yaml --mode protocol $TR $PATHS
  $PY scripts/teacher_traces.py configs/teacher.yaml --mode protocol --val 300 $TR $PATHS
  $PY scripts/teacher_traces.py configs/teacher.yaml --mode rag $TR $PATHS
  $PY scripts/teacher_traces.py configs/teacher.yaml --mode cot $TR $PATHS
}

s1b() {  # SFT: every core branch x seed on the same traces; LoRA vs full FT pilot
  T="data.train=$RUNS/teacher/traces_protocol.jsonl data.val=$RUNS/teacher/traces_val.jsonl $D"
  for B in sealedread nomask daft; do for S in $SEEDS; do
    bash scripts/cluster/train.sh configs/sft.yaml train.branch=$B run.seed=$S run.name=sft_${B}_s$S $T; done; done
  bash scripts/cluster/train.sh configs/sft.yaml train.branch=rag run.name=sft_rag_s0 $D \
    data.train=$RUNS/teacher/traces_rag.jsonl data.val=$RUNS/teacher/traces_val.jsonl
  bash scripts/cluster/train.sh configs/sft.yaml train.branch=sealong run.name=sft_sealong_s0 $D \
    data.train=$RUNS/teacher/traces_cot.jsonl data.val=null
  bash scripts/cluster/train.sh configs/sft.yaml train.branch=sealedread train.full_ft=true run.name=sft_sealedread_full $T
}

s2() {   # one ReST-EM round per branch (seed 0), then the M-step
  for B in sealedread nomask rag; do
    bash scripts/cluster/par.sh rest_$B scripts/rest_em.py configs/rest_em.yaml rest_em.branch=$B \
      rest_em.init_from=$RUNS/sft_${B}_s0 rest_em.samples=$DATA/train.jsonl rest_em.out=$RUNS/rest_$B
    $PY scripts/rest_em.py configs/rest_em.yaml --select-only rest_em.branch=$B rest_em.out=$RUNS/rest_$B \
      rest_em.samples=$DATA/train.jsonl $PATHS
    for S in $SEEDS; do
      bash scripts/cluster/train.sh configs/sft.yaml train.branch=$B run.seed=$S run.name=rest_${B}_s$S $D \
        init_from=$RUNS/sft_${B}_s0 data.train=$RUNS/rest_$B/selected_soft.jsonl \
        data.val=$RUNS/teacher/traces_val.jsonl
    done
  done
}

s4() {   # RQ3: re-filter round-1 samples of SealedRead and redo the M-step
  for V in hard nogate; do
    $PY scripts/rest_em.py configs/rest_em.yaml --select-only rest_em.variant=$V rest_em.out=$RUNS/rest_sealedread \
      rest_em.samples=$DATA/train.jsonl $PATHS
    bash scripts/cluster/train.sh configs/sft.yaml train.branch=sealedread run.name=rq3_$V $D \
      init_from=$RUNS/sft_sealedread_s0 data.train=$RUNS/rest_sealedread/selected_$V.jsonl \
      data.val=$RUNS/teacher/traces_val.jsonl
  done
}

s5() {   # evaluation (table 8); see docs/06-evaluation.md for the full list and the report commands
  for B in sealedread nomask; do for S in $SEEDS; do
    bash scripts/cluster/par.sh eval_${B}_$S scripts/evaluate.py configs/eval.yaml eval.branch=$B eval.tag=${B}_s$S \
      engine.adapter=$RUNS/rest_${B}_s$S/best/adapter eval.out=$RUNS/eval; done; done
  bash scripts/cluster/par.sh eval_plain scripts/evaluate.py configs/eval.yaml eval.branch=plain eval.out=$RUNS/eval
}

"$@"
