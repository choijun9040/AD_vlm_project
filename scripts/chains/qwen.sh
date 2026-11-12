#!/bin/bash
# 개정 30 A — 공식 Qwen, DriveLM, 공식 설정 (이어 하기 판). 각 단계·각 스크립트가 끊긴 곳부터 잇는다.
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/workspace/venv_main/bin/python; O=eval_results/cmp; V=data/drivelm_official/val799.jsonl; C=data/drivelm_official/train100.jsonl
step q_diag $O/dlm_diag_qwen.json -- bash -c "$PY scripts/diag_realworldqa.py --q $V --out $O/dlm_diag_qwen.json >> logs/dlm_diag_qwen.log 2>&1" || exit 1
step q_gen $O/dlm_gen_qwen.json.done -- bash -c "$PY scripts/eval_realworldqa.py --q $V --calib_q $C --runs fp32,bf16,fp16,fp16_resid --max_new_tokens 128 --out $O/dlm_gen_qwen.json >> logs/dlm_gen_qwen.log 2>&1 && touch $O/dlm_gen_qwen.json.done" || exit 1
step q_fid $O/dlm_fid_qwen.json -- bash -c "$PY scripts/fidelity_rwqa.py qwen --q $V --calib_q $C --out $O/dlm_fid_qwen.json >> logs/dlm_fid_qwen.log 2>&1" || exit 1
step q_backbone $O/llm_headroom_official_dlm799.json -- bash -c "$PY scripts/probe_llm_headroom.py --checkpoint none --tag official_dlm799 --limit 799 --attn default --max_pixels 12845056 --min_pixels 3136 --out_dir $O >> logs/dlm_backbone.log 2>&1" || exit 1
touch logs/rev30a_qwen.done; log "사슬 끝"
