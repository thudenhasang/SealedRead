#!/usr/bin/env bash
# Phase B of docs/07-colab-test-plan.md (L4 or A100, about 1-2 hours): the whole chain with Qwen3-0.6B, short
# contexts, about a hundred questions.
# The numbers mean nothing; every script, checkpoint and gate must run without a hand fix.
#   bash scripts/colab/run_rehearsal.sh
source "$(dirname "$0")/common.sh"
S=/content/data/srd_small
R=$RUNS/rehearsal
ensure_shards || exit 1
restore_data $S
# train shards only (s0, s1 for training; s4 plays dev/test): the real s90 dev/test stay unseen (D24)
step reh_build $PY scripts/build_data.py --config configs/data.yaml --shards-dir /content/data/shards --out $S \
  "build.train_shards=[0,1]" build.eval_shard=4 build.tokenizer=Qwen/Qwen3-0.6B build.train.n=120 "build.train.length=[4000,8000]" build.val.n=30 \
  "build.val.length=[4000,8000]" build.reserve.n=20 "build.reserve.length=[4000,8000]" build.dev.n=40 \
  "build.dev.lengths={8192: 40}" build.test.n=40 "build.test.lengths={8192: 40, 16384: 40}" \
  "build.test.fictional_lengths=[8192]" || exit 1
save_data $S
step reh_check $PY scripts/check_data.py $S/*.jsonl --out $S/checks.json || exit 1
T="engine.model=Qwen/Qwen3-0.6B engine.tensor_parallel_size=1 engine.max_model_len=20000 \
  student_tokenizer=Qwen/Qwen3-0.6B teacher.samples=$S/train.jsonl teacher.reserve=$S/reserve.jsonl teacher.total=80 \
  teacher.out=$R/teacher teacher.pilot=20"
step reh_teacher $PY scripts/teacher_traces.py configs/teacher.yaml --mode protocol $T || exit 1
step reh_teacher_val $PY scripts/teacher_traces.py configs/teacher.yaml --mode protocol --val 20 $T || exit 1
# a 0.6B "teacher" passes the trace filter rarely: top up with scripted traces so the later stages have data
for F in traces_protocol:train traces_val:val; do
  N=$(cat "$R/teacher/${F%%:*}.jsonl" 2>/dev/null | wc -l)
  log "${F%%:*}: $N teacher traces"
  [ "$N" -lt 40 ] && $PY scripts/make_scripted_traces.py --samples "$S/${F##*:}.jsonl" --out "$R/teacher/${F%%:*}.jsonl" \
    --n $((40 - N)) --append
done
for B in sealedread nomask; do
  step "reh_sft_$B" $PY scripts/train_sft.py configs/debug_small.yaml train.branch=$B run.name=reh_sft_$B \
    run.out_dir=$R data.train=$R/teacher/traces_protocol.jsonl data.val=$R/teacher/traces_val.jsonl || exit 1
  step "reh_rest_$B" $PY scripts/rest_em.py configs/rest_em.yaml rest_em.branch=$B rest_em.init_from=$R/reh_sft_$B \
    rest_em.samples=$S/train.jsonl rest_em.out=$R/rest_$B engine.model=Qwen/Qwen3-0.6B engine.max_model_len=20000 \
    model.name=Qwen/Qwen3-0.6B || exit 1
  step "reh_mstep_$B" $PY scripts/train_sft.py configs/debug_small.yaml train.branch=$B run.name=reh_rest_$B \
    run.out_dir=$R init_from=$R/reh_sft_$B data.train=$R/rest_$B/selected_soft.jsonl \
    data.val=$R/teacher/traces_val.jsonl || exit 1
  step "reh_eval_$B" $PY scripts/evaluate.py configs/eval.yaml eval.branch=$B engine.model=Qwen/Qwen3-0.6B \
    engine.max_model_len=20000 engine.adapter=$R/reh_rest_$B/best/adapter "eval.files=[$S/test_8k.jsonl,$S/test_16k.jsonl]" \
    eval.out=$R/eval || exit 1
done
$PY scripts/report.py compare --a "$R/eval/sealedread__*.jsonl" --b "$R/eval/nomask__*.jsonl"
rsync -a "$R/" "$DRIVE/runs/rehearsal/"
echo "REHEARSAL_DONE"
