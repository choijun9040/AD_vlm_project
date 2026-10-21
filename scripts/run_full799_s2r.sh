#!/bin/bash
# 평가 데이터 전부로 통일 — 2단계(이어 하기 판): 정확도 11,309문항 × 9조건, 문항별 저장 (사전 등록 4fddeb4)
set -u; source /workspace/vlm_project_2/scripts/full799_lib.sh
FIX=checkpoints/baseline_v2_fixa_native_tower_full799.pt
until [ -f $FIX ] && ! pgrep -f "headroom_guard.py fix --checkpoint checkpoints/student_baseline_v2" >/dev/null; do sleep 60; done
NAT="--max_pixels 1440000 --min_pixels 3136"
ev(){ local m=$1 tag=$2; shift 2
  step "정확도 $m $tag" $O/${m}_epoch_1_${tag}_nuscenesqa_val.json 16 1 $PY scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/$m/epoch_1 \
      --dump_predictions --log_every 1000 --out_dir $O "$@" --tag $tag; }
ev student_baseline_v2 native_fp16       $NAT --lora_dtype float16
ev student_baseline_v2 native_fp16_fixa  $NAT --lora_dtype float16 --vision_state $FIX
ev student_baseline_v2 native_bf16       $NAT --lora_dtype bfloat16
ev student_full        native_fp16       $NAT --lora_dtype float16
ev student_full        native_bf16       $NAT --lora_dtype bfloat16
ev student_baseline_v2 eval_fp16         --lora_dtype float16
ev student_baseline_v2 eval_bf16         --lora_dtype bfloat16
ev student_full        eval_fp16         --lora_dtype float16
ev student_full        eval_bf16         --lora_dtype bfloat16
stage_done s2r
