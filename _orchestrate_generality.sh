#!/bin/bash
# ①②③ — 한계 완화 실험 묶음. λ 스윕과 그 사후 원본 해상도 프로파일(D)이 끝난 뒤 실행.
# 전부 추론만 하며 학습은 없다.
#
#  ③ 학습 궤적    : 5변형 × step_{5000..25000} → "격차가 초기에 형성되어 유지된다"
#                   (λ 스윕의 5,000스텝 제한, 한계 4 완화)
#  ②a 내용 변화·토큰 수 고정 대조군: CAM_BACK (nuScenes, 1600x900이라 토큰 수 동일)
#      → "데이터셋 일반성"이 **아니다**. 토큰 수를 고정한 채 장면 내용만 바꿔,
#        ②b의 차이를 데이터셋 탓과 토큰 수 탓으로 분리할 기준을 만든다.
#  ②b 데이터셋 일반성: CODA-LM 193장 (nuScenes 밖). 720p이라 토큰 수가 달라, nuScenes에서
#      측정한 "토큰 수 → 활성" 곡선 위에 놓이는지로 비교한다.
#  ① 계열 일반성  : 공식 위험 구성 체크포인트 + 다른 계열 (한계 1 부분 완화)
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

WAIT_PID=$1
echo "[GEN] 선행 작업(PID $WAIT_PID) 종료 대기... ($(date '+%F %H:%M:%S'))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
echo "[GEN] 시작 ($(date '+%F %H:%M:%S'))"

run() {  # run <라벨> <명령...>
  local label=$1; shift
  echo "[GEN] $label 시작 ($(date '+%H:%M:%S'))"
  if "$@" >> logs/generality.log 2>&1; then
    echo "[GEN] $label 완료 ($(date '+%H:%M:%S'))"
  else
    echo "[GEN] ERROR: $label 실패 — logs/generality.log 확인"
  fi
}

# ---------- ③ 학습 궤적 ----------
EXTRA=""
for V in student_baseline_v2 student_kd_only_v4 student_temporal student_spatial student_full; do
  for S in 5000 10000 15000 20000 25000; do
    [ -d "checkpoints/$V/step_$S" ] && EXTRA="$EXTRA ${V}@${S}=checkpoints/$V/step_$S"
  done
done
run "③ 학습 궤적" python scripts/profile_vision_activations.py \
  --limit 150 --log_every 0 --only __none__ --extra $EXTRA \
  --out eval_results/training_trajectory_profile.json \
  --dump eval_results/training_trajectory_per_image.jsonl

# ---------- ② 데이터 일반성 ----------
# ②-a 내용 변화·토큰 수 고정 대조군.
#   CAM_BACK은 nuScenes라 원본이 1600x900으로 CAM_FRONT와 동일 → 토큰 수(1,836)가 같다.
#   즉 "장면 내용만" 바꾸는 유일한 조건이며, ②b(CODA-LM)는 내용과 토큰 수가 동시에
#   달라지므로 ②a가 있어야 ②b의 차이를 데이터셋 탓과 토큰 수 탓으로 나눌 수 있다.
#   (CAM_FRONT_LEFT는 CAM_BACK에 추가로 주는 정보가 거의 없어 제외했다.)
run "②a CAM_BACK" python scripts/profile_alt_images.py --camera CAM_BACK --limit 250
# ②-b 데이터셋 일반성 — nuScenes 밖 (CODA-LM, 720p이라 토큰 수가 다르다.
#     nuScenes의 "토큰 수 → 활성" 곡선 위에 놓이는지로 비교한다)
run "②b CODA-LM" python scripts/profile_alt_images.py --image_dir data/codalm_mini --limit 143

# ---------- ① 계열 일반성 ----------
run "① 타 계열" python scripts/profile_other_vlm_families.py --limit 100

echo "[GEN] 전체 완료 $(date '+%F %H:%M:%S')"
