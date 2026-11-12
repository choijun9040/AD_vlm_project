#!/bin/bash
# 개정 20 덧붙임 — FLUX 측정용(101~200번) 임베딩 → 측정만
cd /workspace/vlm_project_2
export HF_HOME=/workspace/.hf_home
bash scripts/run_when_free.sh 30000 logs/flux_emb_calib.done logs/flux_emb_calib.log bash -c "/workspace/venv_sd/bin/python scripts/flux_embed.py --tsv data/PartiPrompts.tsv --start 100 --n 100 --out eval_results/cmp/flux_prompts_calib.pt && touch logs/flux_emb_calib.done"
[ -e logs/flux_emb_calib.done ] || { echo "임베딩 실패"; exit 1; }
bash scripts/run_when_free.sh 40000 logs/flux_calib.done logs/flux_calib.log bash -c "/workspace/venv_sd/bin/python scripts/cmp_flux.py --emb eval_results/cmp/flux_prompts_calib.pt --calib_only --out eval_results/cmp/flux_calib.json && touch logs/flux_calib.done"
echo "사슬 끝"
