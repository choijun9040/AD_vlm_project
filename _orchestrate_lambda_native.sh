#!/bin/bash
# D — λ 스윕 체크포인트를 **원본 해상도**에서 프로파일.
#
# 목적: 인과 사슬의 접합부를 메운다. 링크 ①(λ→활성 크기)은 5,000스텝 모델에서,
# 링크 ②(활성 크기→붕괴)는 완주 모델에서 측정되어 모집단이 다르다. λ 체크포인트를
# 원본 해상도에서 재면 링크 ①의 결과 변수를 링크 ②와 같은 척도(fp16 여유 비율)로
# 둘 수 있어 접합부가 사라진다.
#
# 주의: 실행 중인 _orchestrate_lambda_sweep.sh를 절대 수정하지 않는다(bash는 스크립트를
# 점진적으로 읽으므로 실행 중 편집은 실행을 깨뜨린다). 별도 스크립트로 사후 실행한다.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

WAIT_PID=$1
echo "[NATIVE] λ 스윕(PID $WAIT_PID) 종료 대기... ($(date '+%F %H:%M:%S'))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
echo "[NATIVE] 종료 확인, 시작 ($(date '+%F %H:%M:%S'))"

EXTRA=""
for LAM in 0.0 0.5 1.0 2.0; do
  CKPT="checkpoints/sweep_align${LAM}/step_5000"
  if [ -d "$CKPT" ]; then
    EXTRA="$EXTRA sweep_align${LAM}=${CKPT}"
  else
    echo "[NATIVE] WARNING: $CKPT 없음 — 건너뜀"
  fi
done

if [ -z "$EXTRA" ]; then
  echo "[NATIVE] ERROR: 대상 체크포인트가 하나도 없다"; exit 1
fi

# 원본 해상도(1,440,000px ≈ 1,836 vision token) — 해상도 스윕의 최종 지점과 동일 조건
python scripts/profile_vision_activations.py \
  --limit 250 --log_every 0 \
  --min_pixels 3136 --max_pixels 1440000 \
  --only __none__ --extra $EXTRA \
  --out eval_results/lambda_sweep_native_profile.json \
  --dump eval_results/lambda_sweep_native_per_image.jsonl \
  > logs/lambda_native_profile.log 2>&1

if [ $? -eq 0 ]; then
  echo "[NATIVE] 완료 ($(date '+%F %H:%M:%S')) — eval_results/lambda_sweep_native_profile.json"
else
  echo "[NATIVE] ERROR: 실패 — logs/lambda_native_profile.log 확인"
fi
