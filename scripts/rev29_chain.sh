#!/bin/bash
# 개정 29 — Gemma 4 진단(RealWorldQA 765장) 4조건. 2026-11-04
set -u
cd /workspace/vlm_project_2
PG=/workspace/venv_gemma/bin/python
for m in 26B-A4B 31B; do for t in 280 1120; do
  f=eval_results/cmp/diag_gemma_${m}_$t.json; [ -f $f ] && continue
  echo "[$(date '+%F %T')] $m $t"
  $PG scripts/diag_gemma_rwqa.py --model_dir /workspace/gemma4_vision/gemma-4-$m-it --tokens $t --out $f > logs/rev29_${m}_$t.log 2>&1 || { tail -4 logs/rev29_${m}_$t.log; continue; }
  grep -hE "^\[" logs/rev29_${m}_$t.log | cut -c1-250
done; done
echo "[$(date '+%F %T')] 사슬 끝"
