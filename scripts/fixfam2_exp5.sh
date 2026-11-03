#!/bin/bash
# 사전 등록 개정 8 — 실험 5: 고정 배수 대조. 원격 A100.
#   nohup bash scripts/fixfam2_exp5.sh > logs/fixfam2_exp5.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam2; mkdir -p $O
NAT="--max_pixels 1440000 --min_pixels 3136"
BASE=checkpoints/student_baseline_v2/epoch_1
ARMS="F4 F8 P8 R8"

# ── 팔 만들기·상실률·fp16 오차 (팔 단위로 이어 한다) ──────────────────────────────
step "실험5 Qwen 팔" $O/exp5_qwen_arms.json 40 4 \
  $PY scripts/fixed_mult_arms.py qwen --checkpoint $BASE $NAT --arms ${ARMS// /,} --out $O/exp5_qwen_arms.json
for M in gemma-4-26B-A4B-it gemma-4-31B-it; do for ST in 280 1120; do
  step "실험5 Gemma 4 $M $ST" $O/exp5_gemma_${M}_st${ST}.json 30 1 \
    $PG scripts/fixed_mult_arms.py gemma --model_dir /workspace/gemma4_vision/$M --max_soft_tokens $ST --out $O/exp5_gemma_${M}_st${ST}.json
done; done

# ── 정확도 (11,309문항, 팔마다 8조각, 4개씩) ─────────────────────────────────────
jobs_=()
for A in $ARMS; do for k in 0 1 2 3 4 5 6 7; do jobs_+=("$A:$k"); done; done
run_shard(){ local A=${1%%:*} k=${1##*:}
  step "실험5 정확도 $A 조각 $k/8" $O/student_baseline_v2_epoch_1_native_fp16_exp5_${A}_shard${k}of8_nuscenesqa_val.json 20 1 \
    $PY scripts/eval_nuscenesqa_val.py --checkpoint $BASE --dump_predictions --log_every 1000 --out_dir $O $NAT \
    --lora_dtype float16 --vision_state checkpoints_fix/baseline_v2_exp5_${A}.pt --tag native_fp16_exp5_${A} --shard $k --num_shards 8
}
i=0
for j in "${jobs_[@]}"; do
  run_shard $j > logs/fixfam2_exp5_${j/:/_}.log 2>&1 & sleep 90
  i=$((i+1)); [ $((i % 4)) -eq 0 ] && wait
done
wait
grep -h "시작\|끝\|실패\|포기" logs/fixfam2_exp5_*_[0-7].log
for A in $ARMS; do
  $PY scripts/compare_fix_predictions.py --fix "$O/student_baseline_v2_epoch_1_native_fp16_exp5_${A}_shard*of8_predictions.jsonl" \
    --ref "eval_results/full799/student_baseline_v2_epoch_1_native_bf16_shard*of8_predictions.jsonl" --out $O/exp5_${A}_vs_bf16.json
done
stage_done fixfam2_exp5
