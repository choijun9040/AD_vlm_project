#!/bin/bash
# 실험 5 후속 — P⅛·R⅛ 팔(2026-11-03 18:49 디스크 부족으로 팔 만들기가 포기된 뒤). 사전 등록 개정 8 그대로.
# 실험 5 대기열이 끝난 뒤 돈다. 팔 만들기는 끝난 팔(F¼·F⅛)을 건너뛰고, 정확도는 문항 단위로 이어 한다.
#   nohup bash scripts/fixfam2_exp5b.sh > logs/fixfam2_exp5b.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam2
NAT="--max_pixels 1440000 --min_pixels 3136"
BASE=checkpoints/student_baseline_v2/epoch_1
log "대기: 실험 5 대기열 완료 표시"
until [ -f $O/.stage_fixfam2_exp5_done ]; do sleep 300; done
step "실험5 Qwen 팔 P8·R8" $O/exp5_qwen_arms.json 40 4 \
  $PY scripts/fixed_mult_arms.py qwen --checkpoint $BASE $NAT --arms F4,F8,P8,R8 --out $O/exp5_qwen_arms.json
run_shard(){ local A=${1%%:*} k=${1##*:}
  step "실험5 정확도 $A 조각 $k/8" $O/student_baseline_v2_epoch_1_native_fp16_exp5_${A}_shard${k}of8_nuscenesqa_val.json 20 1 \
    $PY scripts/eval_nuscenesqa_val.py --checkpoint $BASE --dump_predictions --log_every 1000 --out_dir $O $NAT \
    --lora_dtype float16 --vision_state checkpoints_fix/baseline_v2_exp5_${A}.pt --tag native_fp16_exp5_${A} --shard $k --num_shards 8
}
i=0
for A in P8 R8; do for k in 0 1 2 3 4 5 6 7; do
  run_shard $A:$k > logs/fixfam2_exp5b_${A}_$k.log 2>&1 & sleep 90
  i=$((i+1)); [ $((i % 4)) -eq 0 ] && wait
done; done
wait
for A in F4 F8 P8 R8; do
  $PY scripts/compare_fix_predictions.py --fix "$O/student_baseline_v2_epoch_1_native_fp16_exp5_${A}_shard*of8_predictions.jsonl" \
    --ref "eval_results/full799/student_baseline_v2_epoch_1_native_bf16_shard*of8_predictions.jsonl" --out $O/exp5_${A}_vs_bf16.json
done
stage_done fixfam2_exp5b
