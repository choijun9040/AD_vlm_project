#!/bin/bash
# λ_align dose-response 스윕 — 활성 크기 감소가 상관인지 인과인지 가르는 실험.
#
# spatial 변형(λ_temporal=0, temporal_k=1)을 고정하고 lambda_spatial만 0 / 0.5 / 1.0 / 2.0으로
# 바꿔 각각 5,000 optimizer step씩 학습한다. LR 스케줄은 원래의 27,037 스텝 기준을 그대로
# 쓰므로(스크립트가 total_steps를 건드리지 않는다) 기존 student_spatial/step_5000과 같은
# 학습률 궤적 위에 놓인다. 시드를 고정해 데이터 순서·LoRA 초기화도 λ 간에 동일하게 만든다.
#
# 각 학습이 끝날 때마다 곧바로 활성 프로파일을 재서, 전체가 끝나기 전에도 추세를 볼 수 있다.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

SEED=1234
MAX_STEPS=5000
LAMBDAS="0.0 0.5 1.0 2.0"
PROFILE_JSON=eval_results/lambda_sweep_activation_profile.json
PROFILE_JSONL=eval_results/lambda_sweep_activation_per_image.jsonl

echo "[SWEEP] 시작 $(date '+%F %H:%M:%S')  seed=$SEED  max_steps=$MAX_STEPS"

for LAM in $LAMBDAS; do
  TAG="align${LAM}"
  OUT="checkpoints/sweep_${TAG}"
  LOG="logs/sweep_${TAG}.log"

  if [ -d "$OUT/step_${MAX_STEPS}" ]; then
    echo "[SWEEP] $TAG 이미 완료 — 건너뜀"
  else
    echo "[SWEEP] 학습 시작: λ_spatial=$LAM → $OUT  ($(date '+%m-%d %H:%M:%S'))"
    python scripts/train_distillation.py \
      --variant spatial \
      --lambda_spatial "$LAM" \
      --output_dir "$OUT" \
      --max_steps "$MAX_STEPS" \
      --seed "$SEED" \
      > "$LOG" 2>&1
    if [ $? -ne 0 ]; then
      echo "[SWEEP] ERROR: $TAG 학습 실패 — $LOG 확인. 다음 λ로 넘어감"
      continue
    fi
    echo "[SWEEP] 학습 완료: $TAG  ($(date '+%m-%d %H:%M:%S'))"
  fi

  echo "[SWEEP] 활성 프로파일 측정: $TAG"
  python scripts/profile_vision_activations.py \
    --only __none__ \
    --extra "sweep_${TAG}=${OUT}/step_${MAX_STEPS}" \
    --out "$PROFILE_JSON" --dump "$PROFILE_JSONL" \
    >> logs/sweep_profile.log 2>&1 \
    && echo "[SWEEP] 프로파일 완료: $TAG" \
    || echo "[SWEEP] ERROR: $TAG 프로파일 실패 — logs/sweep_profile.log 확인"
done

echo "[SWEEP] 전체 완료 $(date '+%F %H:%M:%S')"
echo "[SWEEP] 결과: $PROFILE_JSON"
