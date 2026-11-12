#!/bin/bash
# (R) 자동 도구 회귀 사슬 — 해결기를 고칠 때마다 네 시험을 차례로(2026-11-04). 각 시험은 GPU 여유를 기다린다.
cd /workspace/vlm_project_2
T=$(date +%m%d%H%M)
bash scripts/run_when_free.sh 15000 logs/ar_gemma_$T.done logs/ar_gemma_$T.log bash -c "/workspace/venv_gemma/bin/python scripts/test_auto_resid_towers.py gemma && touch logs/ar_gemma_$T.done"
bash scripts/run_when_free.sh 20000 logs/ar_vae_$T.done logs/ar_vae_$T.log bash -c "/workspace/venv_sd/bin/python scripts/test_auto_resid_vae.py && touch logs/ar_vae_$T.done"
bash scripts/run_when_free.sh 15000 logs/ar_qwen_off_$T.done logs/ar_qwen_off_$T.log bash -c "/workspace/venv_main/bin/python scripts/test_auto_resid_towers.py qwen none && touch logs/ar_qwen_off_$T.done"
bash scripts/run_when_free.sh 15000 logs/ar_qwen_stu_$T.done logs/ar_qwen_stu_$T.log bash -c "/workspace/venv_main/bin/python scripts/test_auto_resid_towers.py qwen checkpoints/student_baseline_v2/epoch_1 && touch logs/ar_qwen_stu_$T.done"
echo "사슬 끝 $T"
