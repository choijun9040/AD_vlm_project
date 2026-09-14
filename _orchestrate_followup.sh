#!/bin/bash
# B(정렬 대상 ablation) 종료 후 남은 극복 항목을 순차 실행한다.
#
#  ⑤ fixA 정확도 정밀도 — 300문항(노이즈 ±0.7pp) → 전체 11,309문항.
#     "정확도 무손실" 주장을 ±0.1pp 수준으로 확정한다. fixB는 no-op으로 이미 확인돼 제외.
#  ⑥ 6장 함의 — 위험 구성 체크포인트를 추가로 프로파일한다. 실제로 여유가 없는 사례가
#     하나라도 나오면 주장이 회복되고, 안 나오면 "구성만으로는 판단 불가"로 정직하게
#     축소한다. 디스크가 20GB뿐이라 하나씩 받고 지운다(--cleanup).
#  ④ λ 용량-반응 분산 — λ=1.0을 다른 시드로 1회 반복해 지점당 n=1 제약을 완화한다.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

WAIT_PID=$1
echo "[FOLLOW] B(PID $WAIT_PID) 종료 대기... ($(date '+%F %H:%M:%S'))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
echo "[FOLLOW] 시작 ($(date '+%F %H:%M:%S'))"

run() {
  local label=$1; shift
  echo "[FOLLOW] $label 시작 ($(date '+%H:%M:%S'))"
  if "$@" >> logs/followup.log 2>&1; then
    echo "[FOLLOW] $label 완료 ($(date '+%H:%M:%S'))"
  else
    echo "[FOLLOW] ERROR: $label 실패 — logs/followup.log 확인"
  fi
}

# ---------- ⑤ fixA 전체 평가 ----------
run "⑤ fixA 전체 평가(11,309문항)" \
  python scripts/probe_posthoc_rescale.py --scale 3 \
    --n_images 250 --n_qa 11309 --conds none,fixa \
    --out eval_results/posthoc_rescale_full.json

# ---------- ⑥ 위험 구성 추가 표본 ----------
run "⑥ 위험 구성 추가 프로파일" \
  python scripts/profile_other_vlm_families.py --limit 100 --cleanup

# ---------- ④ λ=1.0 다른 시드 ----------
run "④ λ=1.0 seed 4321 학습" \
  python scripts/train_distillation.py --variant spatial \
    --lambda_align 1.0 --output_dir checkpoints/sweep_align1.0_seed4321 \
    --max_steps 5000 --seed 4321
if [ -d checkpoints/sweep_align1.0_seed4321/step_5000 ]; then
  run "④ 프로파일" \
    python scripts/profile_vision_activations.py --limit 250 --log_every 0 \
      --only __none__ \
      --extra "sweep_align1.0_seed4321=checkpoints/sweep_align1.0_seed4321/step_5000" \
      --out eval_results/lambda_sweep_activation_profile.json \
      --dump eval_results/lambda_sweep_activation_per_image.jsonl
fi

echo "[FOLLOW] 전체 완료 $(date '+%F %H:%M:%S')"
