#!/bin/bash
# 세션 전용 오케스트레이션 스크립트 (파이프라인 일부 아님, git 추적 안 함)
# 기존 _orchestrate_v2.sh가 실행 중인 kd_only_v2(PID 4176157) 완료를 별도로
# 기다린 뒤, train_distillation.py --variant spatial -> --variant temporal
# 순으로 이어서 실행한다 (실행 중인 스크립트 파일을 직접 수정하지 않기 위해
# 새 오케스트레이터를 분리했다).
set -uo pipefail

cd /workspace/vlm_project_2
source /venv/main/bin/activate

KD_PID=4176157

echo "[ORCHESTRATOR-ABL] kd_only_v2(PID $KD_PID) 완료 대기 중... ($(date '+%Y-%m-%d %H:%M:%S'))"
while kill -0 "$KD_PID" 2>/dev/null; do
  sleep 60
done

if grep -q "KD only Fine-tuning 완료" logs/kd_only_v2.log; then
  echo "[ORCHESTRATOR-ABL] kd_only_v2 정상 완료 확인됨 ($(date '+%Y-%m-%d %H:%M:%S'))"
else
  echo "[ORCHESTRATOR-ABL] WARNING: kd_only_v2 프로세스 종료됐지만 정상 완료 메시지를 로그에서 못 찾음 - 확인 필요 ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

echo "[ORCHESTRATOR-ABL] spatial ablation 학습 시작 ($(date '+%Y-%m-%d %H:%M:%S'))"
python scripts/train_distillation.py --variant spatial > logs/spatial.log 2>&1 &
SPATIAL_PID=$!
echo $SPATIAL_PID > logs/spatial.pid
wait $SPATIAL_PID
SPATIAL_EXIT=$?
if [ "$SPATIAL_EXIT" -eq 0 ] && grep -q "Stage 3 Distillation 완료" logs/spatial.log; then
  echo "[ORCHESTRATOR-ABL] spatial 정상 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
else
  echo "[ORCHESTRATOR-ABL] ERROR: spatial 비정상 종료 (exit=$SPATIAL_EXIT) ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

echo "[ORCHESTRATOR-ABL] temporal ablation 학습 시작 ($(date '+%Y-%m-%d %H:%M:%S'))"
python scripts/train_distillation.py --variant temporal > logs/temporal.log 2>&1 &
TEMPORAL_PID=$!
echo $TEMPORAL_PID > logs/temporal.pid
wait $TEMPORAL_PID
TEMPORAL_EXIT=$?
if [ "$TEMPORAL_EXIT" -eq 0 ] && grep -q "Stage 3 Distillation 완료" logs/temporal.log; then
  echo "[ORCHESTRATOR-ABL] temporal 정상 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
else
  echo "[ORCHESTRATOR-ABL] ERROR: temporal 비정상 종료 (exit=$TEMPORAL_EXIT) ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

echo "[ORCHESTRATOR-ABL] ablation 파이프라인(spatial -> temporal) 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
