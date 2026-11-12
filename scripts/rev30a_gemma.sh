#!/bin/bash
# 개정 30 A — Gemma 4, DriveLM(val 799·train 100), 공식 처리기(280 기본·1120 최대). 2026-11-04
set -u
cd /workspace/vlm_project_2
PG=/workspace/venv_gemma/bin/python; O=eval_results/cmp; V=data/drivelm_official/val799.jsonl; C=data/drivelm_official/train100.jsonl
log(){ echo "[$(date '+%F %T')] $*"; }
for m in 26B-A4B 31B; do for t in 280 1120; do
  D=/workspace/gemma4_vision/gemma-4-$m-it
  log "진단 $m $t"; [ -f $O/dlm_diag_gemma_${m}_$t.json ] || $PG scripts/diag_gemma_rwqa.py --model_dir $D --tokens $t --q $V --out $O/dlm_diag_gemma_${m}_$t.json > logs/dlm_diag_gemma_${m}_$t.log 2>&1
  grep -hE "^\[" logs/dlm_diag_gemma_${m}_$t.log | cut -c1-250
  log "보존 $m $t"; [ -f $O/dlm_fid_gemma_${m}_$t.json ] || $PG scripts/fidelity_rwqa.py gemma --model_dir $D --tokens $t --q $V --calib_q $C --out $O/dlm_fid_gemma_${m}_$t.json > logs/dlm_fid_gemma_${m}_$t.log 2>&1
  grep -hE "^\[" logs/dlm_fid_gemma_${m}_$t.log | cut -c1-300
done; done
log "사슬 끝"
