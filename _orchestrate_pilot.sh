#!/bin/bash
# 세션 전용 — INT4 평가(기여 3) 완료 후 파일럿(평가 교란 가설) 자동 실행
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

AWQ_PID=$1
echo "[PILOT-ORCH] INT4 평가(PID $AWQ_PID) 완료 대기... ($(date '+%m-%d %H:%M:%S'))"
while kill -0 "$AWQ_PID" 2>/dev/null; do sleep 120; done

if grep -q "전체 완료" logs/orchestrate_awq.log; then
  echo "[PILOT-ORCH] INT4 정상 완료 확인 ($(date '+%m-%d %H:%M:%S'))"
else
  echo "[PILOT-ORCH] WARNING: INT4가 완료 메시지 없이 종료 — 로그 확인 필요"
fi

echo "[PILOT-ORCH] 파일럿 시작 ($(date '+%m-%d %H:%M:%S'))"
python scripts/pilot_metric_confound.py --limit 1000 > logs/pilot_metric_confound.log 2>&1
if [ $? -eq 0 ]; then
  echo "[PILOT-ORCH] 파일럿 완료 ($(date '+%m-%d %H:%M:%S'))"
else
  echo "[PILOT-ORCH] ERROR: 파일럿 실패 — logs/pilot_metric_confound.log 확인"
fi
