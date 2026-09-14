#!/bin/bash
# B 재시도 — ① 재실행이 끝난 뒤 순차로.
set -uo pipefail
cd /workspace/vlm_project_2
source /venv/main/bin/activate
WAIT_PID=$1
echo "[ALIGNTGT2] 선행(PID $WAIT_PID) 대기... ($(date '+%F %H:%M:%S'))"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done

echo "[ALIGNTGT2] 스모크(2스텝) ($(date '+%H:%M:%S'))"
python scripts/train_align_target_ablation.py --target base3b --max_steps 2 \
  --output_dir /tmp/aligntgt_smoke > logs/aligntgt_smoke2.log 2>&1
if [ $? -ne 0 ]; then
  echo "[ALIGNTGT2] ERROR: 스모크 실패 — logs/aligntgt_smoke2.log 확인"; exit 1
fi
rm -rf /tmp/aligntgt_smoke
echo "[ALIGNTGT2] 스모크 통과, 본 실행 5,000스텝 ($(date '+%H:%M:%S'))"

python scripts/train_align_target_ablation.py --target base3b --max_steps 5000 --seed 1234 \
  > logs/aligntgt_base3b.log 2>&1 \
  && echo "[ALIGNTGT2] 학습 완료 ($(date '+%H:%M:%S'))" \
  || { echo "[ALIGNTGT2] ERROR: 학습 실패"; exit 1; }

python scripts/profile_vision_activations.py --limit 250 --log_every 0 \
  --only __none__ --extra "aligntgt_base3b=checkpoints/aligntgt_base3b/step_5000" \
  --out eval_results/align_target_profile.json \
  --dump eval_results/align_target_per_image.jsonl \
  >> logs/aligntgt_base3b.log 2>&1 \
  && echo "[ALIGNTGT2] 전체 완료 ($(date '+%F %H:%M:%S'))" \
  || echo "[ALIGNTGT2] ERROR: 프로파일 실패"
