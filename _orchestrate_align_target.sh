#!/bin/bash
# B — 정렬 대상 ablation. λ 스윕이 끝난 뒤 실행한다(GPU 경합 회피 + train_distillation.py
# 호출이 겹치지 않게).
#
# 먼저 2스텝 스모크로 우회 패치가 동작하는지 확인하고, 통과하면 5,000스텝 본 실행.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate

WAIT_PID=$1
echo "[ALIGNTGT] λ 스윕(PID $WAIT_PID) 종료 대기... ($(date '+%F %H:%M:%S'))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 300; done
echo "[ALIGNTGT] 시작 ($(date '+%F %H:%M:%S'))"

echo "[ALIGNTGT] 스모크(2스텝)"
python scripts/train_align_target_ablation.py --target base3b --max_steps 2 \
  --output_dir /tmp/aligntgt_smoke > logs/aligntgt_smoke.log 2>&1
if [ $? -ne 0 ]; then
  echo "[ALIGNTGT] ERROR: 스모크 실패 — logs/aligntgt_smoke.log 확인. 본 실행 중단"
  exit 1
fi
rm -rf /tmp/aligntgt_smoke
echo "[ALIGNTGT] 스모크 통과"

echo "[ALIGNTGT] 본 실행: base3b 5,000스텝 ($(date '+%H:%M:%S'))"
python scripts/train_align_target_ablation.py --target base3b --max_steps 5000 --seed 1234 \
  > logs/aligntgt_base3b.log 2>&1
if [ $? -ne 0 ]; then
  echo "[ALIGNTGT] ERROR: 학습 실패 — logs/aligntgt_base3b.log 확인"; exit 1
fi
echo "[ALIGNTGT] 학습 완료 ($(date '+%H:%M:%S'))"

echo "[ALIGNTGT] 활성 프로파일"
python scripts/profile_vision_activations.py --limit 250 --log_every 0 \
  --only __none__ --extra "aligntgt_base3b=checkpoints/aligntgt_base3b/step_5000" \
  --out eval_results/align_target_profile.json \
  --dump eval_results/align_target_per_image.jsonl \
  >> logs/aligntgt_base3b.log 2>&1 \
  && echo "[ALIGNTGT] 완료 ($(date '+%F %H:%M:%S'))" \
  || echo "[ALIGNTGT] ERROR: 프로파일 실패"
