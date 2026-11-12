#!/bin/bash
# 개정 21 — 측정용 100장(DriveLM train)으로 Qwen·Gemma c 결정, 6조건 차례로
cd /workspace/vlm_project_2
O=eval_results/cmp
run() { bash scripts/run_when_free.sh 15000 logs/c100_$1.done logs/c100_$1.log bash -c "$2 && touch logs/c100_$1.done"; }
run qwen_off "/workspace/venv_main/bin/python scripts/calib_train100.py qwen none --out $O/calib100_qwen_official.json"
run qwen_stu "/workspace/venv_main/bin/python scripts/calib_train100.py qwen checkpoints/student_baseline_v2/epoch_1 --out $O/calib100_qwen_student.json"
for m in 26B-A4B 31B; do for t in 280 1120; do
  run gemma_${m}_$t "/workspace/venv_gemma/bin/python scripts/calib_train100.py gemma /workspace/gemma4_vision/gemma-4-$m-it --tokens $t --out $O/calib100_gemma_${m}_$t.json"
done; done
echo "사슬 끝"
