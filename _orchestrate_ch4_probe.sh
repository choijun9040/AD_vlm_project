#!/bin/bash
# 4장 보강 — KD 온도 × 어휘 크기 스윕. INT8 작업이 끝난 뒤 실행해 GPU 3중 경합을 피한다.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

WAIT_PID=$1
echo "[CH4] 앞 작업(PID $WAIT_PID) 종료 대기... ($(date '+%F %H:%M:%S'))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
echo "[CH4] 시작 ($(date '+%F %H:%M:%S'))"

python scripts/probe_kd_temperature_vocab.py --n_samples 40 \
  > logs/kd_temperature_vocab.log 2>&1
if [ $? -eq 0 ]; then
  echo "[CH4] 완료 ($(date '+%F %H:%M:%S')) — eval_results/kd_temperature_vocab_sweep.json"
else
  echo "[CH4] ERROR: 실패 — logs/kd_temperature_vocab.log 확인"
fi
