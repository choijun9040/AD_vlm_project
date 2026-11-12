#!/bin/bash
# 개정 30 A — 공식 Qwen2.5-VL-3B, DriveLM(val 799·train 100), 공식 저장소 설정. 2026-11-04
set -u
cd /workspace/vlm_project_2
PY=/workspace/venv_main/bin/python; O=eval_results/cmp; V=data/drivelm_official/val799.jsonl; C=data/drivelm_official/train100.jsonl
log(){ echo "[$(date '+%F %T')] $*"; }
log "진단"; [ -f $O/dlm_diag_qwen.json ] || $PY scripts/diag_realworldqa.py --q $V --out $O/dlm_diag_qwen.json > logs/dlm_diag_qwen.log 2>&1
grep -hE "^\[" logs/dlm_diag_qwen.log | grep -v awq
log "생성 4형식"; [ -f $O/dlm_gen_qwen.json.done ] || { $PY scripts/eval_realworldqa.py --q $V --calib_q $C --runs fp32,bf16,fp16,fp16_resid --max_new_tokens 128 --out $O/dlm_gen_qwen.json > logs/dlm_gen_qwen.log 2>&1 && touch $O/dlm_gen_qwen.json.done; }
grep -hE "^\[" logs/dlm_gen_qwen.log | grep -v awq | cut -c1-300
log "비전 인코더 보존"; [ -f $O/dlm_fid_qwen.json ] || $PY scripts/fidelity_rwqa.py qwen --q $V --calib_q $C --out $O/dlm_fid_qwen.json > logs/dlm_fid_qwen.log 2>&1
grep -hE "^\[" logs/dlm_fid_qwen.log | grep -v awq | cut -c1-300
log "백본"; [ -f $O/llm_headroom_official_dlm799.json ] || $PY scripts/probe_llm_headroom.py --checkpoint none --tag official_dlm799 --limit 799 --attn default --max_pixels 12845056 --min_pixels 3136 --out_dir $O > logs/dlm_backbone.log 2>&1
grep -hE "^\[크기|^\[상실" logs/dlm_backbone.log
log "사슬 끝"
