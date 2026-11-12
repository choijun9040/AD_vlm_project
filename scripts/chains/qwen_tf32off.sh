#!/bin/bash
# 개정 40 — Qwen 비전 인코더 보존(+같은 228장)·문장 일치를 TF32 끄고 다시. 결과는 eval_results/cmp/env4/. 2026-11-10
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PY=/workspace/venv_main/bin/python; O=eval_results/cmp; E=$O/env4; mkdir -p $E
V=data/drivelm_official/val799.jsonl; C=data/drivelm_official/train100.jsonl
while pgrep -f "[f]lux_attn_diag.py" >/dev/null; do sleep 60; done          # FLUX 원인 찾기(fp32 48 GB)가 끝난 뒤 — GPU 메모리
step q40_fid $E/dlm_fid_qwen.json -- bash -c "$PY scripts/fidelity_rwqa.py qwen --forms bf16,fp16,fp16_resid --per_image --no_tf32 --q $V --calib_q $C --out $E/dlm_fid_qwen.json >> logs/env4.log 2>&1" || exit 1
step q40_gen $E/dlm_gen_qwen.json.done -- bash -c "$PY scripts/eval_realworldqa.py --q $V --calib_q $C --runs fp32,bf16,fp16,fp16_resid --max_new_tokens 128 --no_tf32 --out $E/dlm_gen_qwen.json >> logs/env4.log 2>&1 && touch $E/dlm_gen_qwen.json.done" || exit 1
touch logs/env4.done; log "사슬 끝"
