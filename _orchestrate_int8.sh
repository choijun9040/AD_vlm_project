#!/bin/bash
# 해상도 스윕이 끝난 뒤 INT8 활성 양자화 오차 프로파일을 이어서 돌린다.
# GPU를 λ 스윕 학습과만 공유하도록(3중 경합 회피) 앞 작업 종료를 기다린다.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

WAIT_PID=$1
echo "[INT8] 해상도 스윕(PID $WAIT_PID) 종료 대기... ($(date '+%F %H:%M:%S'))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
echo "[INT8] 앞 작업 종료 확인, 시작 ($(date '+%F %H:%M:%S'))"

python scripts/profile_int8_quant_error.py --limit 200 --log_every 100 \
  > logs/int8_quant_error.log 2>&1
if [ $? -eq 0 ]; then
  echo "[INT8] 완료 ($(date '+%F %H:%M:%S')) — eval_results/int8_quant_error.json"
else
  echo "[INT8] ERROR: 실패 — logs/int8_quant_error.log 확인"
fi
