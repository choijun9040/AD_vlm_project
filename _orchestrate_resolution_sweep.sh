#!/bin/bash
# 그림 2용 — 입력 해상도 × fp16 붕괴율 측정.
#
# nuScenes CAM_FRONT 원본이 1600x900(1,440,000px)이라 max_pixels는 그 지점에서 포화한다.
# 따라서 128 / 256 / 512 / 1024 / 1836 vision token에 해당하는 5개 지점만 의미가 있다
# (256이 학습·평가에 쓴 값, 1836이 AWQ 저장본 processor의 기본값에서 실제로 적용되던 값).
#
# λ 스윕 학습과 GPU를 공유하므로 이미지 수를 250장으로 제한한다.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

N_IMAGES=250
MODELS="base_3b student_baseline_v2 student_kd_only_v4 student_temporal student_spatial student_full"
OUTDIR=eval_results/resolution_sweep

echo "[RES] 시작 $(date '+%F %H:%M:%S')  이미지 ${N_IMAGES}장"

for MP in 100352 200704 401408 802816 1440000; do
  OUT="$OUTDIR/res_${MP}.json"
  if [ -f "$OUT" ]; then
    echo "[RES] max_pixels=$MP 이미 완료 — 건너뜀"; continue
  fi
  echo "[RES] max_pixels=$MP 측정 시작 ($(date '+%m-%d %H:%M:%S'))"
  python scripts/profile_vision_activations.py \
    --limit "$N_IMAGES" --log_every 0 \
    --min_pixels 3136 --max_pixels "$MP" \
    --only $MODELS \
    --out "$OUT" \
    --dump "$OUTDIR/res_${MP}_per_image.jsonl" \
    >> logs/resolution_sweep.log 2>&1 \
    && echo "[RES] max_pixels=$MP 완료 ($(date '+%m-%d %H:%M:%S'))" \
    || echo "[RES] ERROR: max_pixels=$MP 실패 — logs/resolution_sweep.log 확인"
done

echo "[RES] 전체 완료 $(date '+%F %H:%M:%S')"
