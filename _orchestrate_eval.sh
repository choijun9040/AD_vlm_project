#!/bin/bash
# 세션 전용 오케스트레이션 스크립트 (파이프라인 일부 아님, git 추적 안 함)
# 5-way ablation 비교를 위해 NuScenes-QA val(11,309 샘플) 평가를
# student_full -> student_spatial -> student_temporal -> student_baseline_v2 ->
# student_kd_only_v2 순으로 순차 실행 (GPU 1개뿐이라 동시 실행 불가).
# 체크포인트당 스모크 테스트 실측 약 1.1s/sample -> 약 3.5시간, 5개 총 약 17~18시간 예상.
set -uo pipefail

cd /workspace/vlm_project_2
source /venv/main/bin/activate

run_eval() {
  local tag="$1"
  local ckpt="$2"
  echo "[ORCHESTRATOR-EVAL] ${tag} 평가 시작 (${ckpt}) ($(date '+%Y-%m-%d %H:%M:%S'))"
  python scripts/eval_nuscenesqa_val.py --checkpoint "$ckpt" > "logs/eval_${tag}.log" 2>&1
  if [ $? -eq 0 ] && grep -q "결과 저장:" "logs/eval_${tag}.log"; then
    echo "[ORCHESTRATOR-EVAL] ${tag} 정상 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
  else
    echo "[ORCHESTRATOR-EVAL] ERROR: ${tag} 비정상 종료 ($(date '+%Y-%m-%d %H:%M:%S'))"
  fi
}

run_eval "full"        "checkpoints/student_full/epoch_1"
run_eval "spatial"      "checkpoints/student_spatial/epoch_1"
run_eval "temporal"     "checkpoints/student_temporal/epoch_1"
run_eval "baseline_v2"  "checkpoints/student_baseline_v2/epoch_1"
run_eval "kd_only_v2"   "checkpoints/student_kd_only_v2/epoch_1"

echo "[ORCHESTRATOR-EVAL] 5-way NuScenes-QA val 평가 전체 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
