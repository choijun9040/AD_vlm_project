#!/bin/bash
# 평가 데이터 전부로 통일 — 2단계: 정확도 NuScenes-QA 검증 11,309문항 전부 × 9개 조건, 문항별 저장 (사전 등록 4fddeb4)
# 교정 조건은 799장으로 새로 푼 가중치(baseline_v2_fixa_native_tower_full799.pt)를 쓴다.
set -u
cd /workspace/vlm_project_2; export HF_HOME=/workspace/hf_home
PY=/workspace/venv_main/bin/python; O=eval_results/full799; mkdir -p $O
FIX=checkpoints/baseline_v2_fixa_native_tower_full799.pt
log(){ echo "[$(date '+%F %T')] $*"; }
ev(){ log "시작 $1 ($2)"; $PY scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/$1/epoch_1 --dump_predictions --log_every 1000 \
      --out_dir $O "${@:3}" --tag $2 2>&1 | grep -E "Overall|exist|count|object|status|comparison|결과 저장|Error|Traceback|processor\]" | tail -9; log "끝 $1 $2"; }
until [ -f $FIX ] && ! pgrep -f "headroom_guard.py fix" >/dev/null; do sleep 60; done
NAT="--max_pixels 1440000 --min_pixels 3136"
ev student_baseline_v2 native_fp16       $NAT --lora_dtype float16
ev student_baseline_v2 native_fp16_fixa  $NAT --lora_dtype float16 --vision_state $FIX
ev student_baseline_v2 native_bf16       $NAT --lora_dtype bfloat16
ev student_full        native_fp16       $NAT --lora_dtype float16
ev student_full        native_bf16       $NAT --lora_dtype bfloat16
ev student_baseline_v2 eval_fp16         --lora_dtype float16
ev student_baseline_v2 eval_bf16         --lora_dtype bfloat16
ev student_full        eval_fp16         --lora_dtype float16
ev student_full        eval_bf16         --lora_dtype bfloat16
log "2단계 완료"
