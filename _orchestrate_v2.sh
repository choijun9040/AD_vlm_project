#!/bin/bash
# 세션 전용 오케스트레이션 스크립트 (파이프라인 일부 아님, git 추적 안 함)
# Full(PID 1050138, 동적 패딩+projection 최종 버전) 완료를 기다린 뒤
# baseline_v2 -> kd_only_v2 순으로 실행 (vision attn LoRA 통일 재학습).
set -uo pipefail

cd /workspace/vlm_project_2
source /venv/main/bin/activate

FULL_PID=1376349

echo "[ORCHESTRATOR] Full(PID $FULL_PID) 완료 대기 중... ($(date '+%Y-%m-%d %H:%M:%S'))"
while kill -0 "$FULL_PID" 2>/dev/null; do
  sleep 60
done

if grep -q "Stage 3 Distillation 완료" logs/full.log; then
  echo "[ORCHESTRATOR] Full 정상 완료 확인됨 ($(date '+%Y-%m-%d %H:%M:%S'))"
else
  echo "[ORCHESTRATOR] WARNING: Full 프로세스 종료됐지만 정상 완료 메시지를 로그에서 못 찾음 - 확인 필요 ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

echo "[ORCHESTRATOR] baseline_v2 학습 시작 ($(date '+%Y-%m-%d %H:%M:%S'))"
python scripts/train_baseline.py > logs/baseline_v2.log 2>&1 &
BASELINE_PID=$!
echo $BASELINE_PID > logs/baseline_v2.pid
wait $BASELINE_PID
BASELINE_EXIT=$?
if [ "$BASELINE_EXIT" -eq 0 ] && grep -q "Baseline Fine-tuning 완료" logs/baseline_v2.log; then
  echo "[ORCHESTRATOR] baseline_v2 정상 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
else
  echo "[ORCHESTRATOR] ERROR: baseline_v2 비정상 종료 (exit=$BASELINE_EXIT) ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

echo "[ORCHESTRATOR] kd_only_v2 학습 시작 ($(date '+%Y-%m-%d %H:%M:%S'))"
python scripts/train_kd_only.py > logs/kd_only_v2.log 2>&1 &
KD_PID=$!
echo $KD_PID > logs/kd_only_v2.pid
wait $KD_PID
KD_EXIT=$?
if [ "$KD_EXIT" -eq 0 ] && grep -q "KD only Fine-tuning 완료" logs/kd_only_v2.log; then
  echo "[ORCHESTRATOR] kd_only_v2 정상 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
else
  echo "[ORCHESTRATOR] ERROR: kd_only_v2 비정상 종료 (exit=$KD_EXIT) ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

echo "[ORCHESTRATOR] 전체 파이프라인(Full -> baseline_v2 -> kd_only_v2) 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
