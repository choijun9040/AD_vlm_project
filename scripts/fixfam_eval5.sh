#!/bin/bash
# 기준 ⑤만 병렬로 — 교정 확장 대기열(fixfam_queue.sh)과 따로, 4조각씩 동시에 (2026-11-01 사용자 요청)
# 같은 step()과 결과 폴더를 쓰므로, 본 대기열이 ⑤에 이르면 끝난 조각은 완료 표시로 건너뛰고 도는 조각은 끝나기를 기다린다.
#   nohup bash scripts/fixfam_eval5.sh > logs/fixfam_eval5.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam; NAT="--max_pixels 1440000 --min_pixels 3136"
if ! $PY -c "import json,sys; d=json.load(open('$O/qwen25_baseline.json')); sys.exit(0 if d.get('residual',{}).get('accepted') and not d.get('applied') else 1)"; then
  log "기준 ⑤ 건너뜀 — 기준 모델에 잔차 재조정만으로 교정되지 않았다(사전 등록: H1이 성립할 때만)"; exit 0
fi
shard(){
  local k=$1
  step "정확도 잔차 재조정 조각 $k/8" $O/student_baseline_v2_epoch_1_native_fp16_resid_shard${k}of8_nuscenesqa_val.json 20 1 \
    $PY scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --dump_predictions --log_every 1000 \
    --out_dir $O $NAT --lora_dtype float16 --vision_state checkpoints_fix/baseline_v2_residual_native.pt \
    --tag native_fp16_resid --shard $k --num_shards 8
}
for grp in "0 1 2 3" "4 5 6 7"; do
  for k in $grp; do shard $k > logs/fixfam_eval5_shard$k.log 2>&1 & sleep 90; done   # 순차로 올려 GPU 여유 확인이 겹치지 않게
  wait
done
grep -h "시작\|끝\|실패\|포기" logs/fixfam_eval5_shard*.log
$PY scripts/compare_fix_predictions.py --fix "$O/student_baseline_v2_epoch_1_native_fp16_resid_shard*of8_predictions.jsonl" \
  --ref "eval_results/full799/student_baseline_v2_epoch_1_native_bf16_shard*of8_predictions.jsonl" --out $O/criterion5.json
stage_done fixfam_eval5
