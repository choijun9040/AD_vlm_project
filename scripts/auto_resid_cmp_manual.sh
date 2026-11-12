#!/bin/bash
# 개정 19 — 자동 구성 대 손 구성 대조(공식 Qwen·학생 Qwen·공식 Gemma)
cd /workspace/vlm_project_2
T=$(date +%m%d%H%M)
bash scripts/run_when_free.sh 15000 logs/am_qwen_off_$T.done logs/am_qwen_off_$T.log bash -c "/workspace/venv_main/bin/python scripts/test_auto_resid_towers.py qwen none && touch logs/am_qwen_off_$T.done"
bash scripts/run_when_free.sh 15000 logs/am_qwen_stu_$T.done logs/am_qwen_stu_$T.log bash -c "/workspace/venv_main/bin/python scripts/test_auto_resid_towers.py qwen checkpoints/student_baseline_v2/epoch_1 && touch logs/am_qwen_stu_$T.done"
bash scripts/run_when_free.sh 15000 logs/am_gemma_$T.done logs/am_gemma_$T.log bash -c "/workspace/venv_gemma/bin/python scripts/test_auto_resid_towers.py gemma && touch logs/am_gemma_$T.done"
echo "사슬 끝 $T"
