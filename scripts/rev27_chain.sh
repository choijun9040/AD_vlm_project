#!/bin/bash
# 개정 27 — 원래 출력(fp32) 보존: Qwen fp32 생성·일치, Qwen·Gemma 비전 인코더 fp32 대비 오차. 2026-11-04
set -u
cd /workspace/vlm_project_2
PY=/workspace/venv_main/bin/python; PG=/workspace/venv_gemma/bin/python; O=eval_results/cmp
log(){ echo "[$(date '+%F %T')] $*"; }
until [ -f logs/rwqa_eval3.done ]; do sleep 120; done
log "정확도 2단계 끝 — fp32 생성"
[ -f $O/rwqa_eval_fp32.json ] || $PY scripts/eval_realworldqa.py --runs fp32 --c 0.25 --out $O/rwqa_eval_fp32.json > logs/rev27_fp32gen.log 2>&1 || { tail -5 logs/rev27_fp32gen.log; exit 1; }
$PY scripts/merge_rwqa_fp32.py --main $O/rwqa_eval_official.json --fp32 $O/rwqa_eval_fp32.json --out $O/rwqa_agree_fp32.json 2>&1 | grep -v awq_compat
log "Qwen 비전 인코더 보존"
[ -f $O/fid_qwen.json ] || $PY scripts/fidelity_rwqa.py qwen --out $O/fid_qwen.json > logs/rev27_fid_qwen.log 2>&1 || { tail -5 logs/rev27_fid_qwen.log; exit 1; }
grep -hE "^\[" logs/rev27_fid_qwen.log | grep -v awq | cut -c1-300
for m in 26B-A4B 31B; do for t in 280 1120; do
  f=$O/fid_gemma_${m}_$t.json; [ -f $f ] && continue
  log "Gemma $m $t"
  $PG scripts/fidelity_rwqa.py gemma --model_dir /workspace/gemma4_vision/gemma-4-$m-it --tokens $t --out $f > logs/rev27_fid_gemma_${m}_$t.log 2>&1 || { tail -5 logs/rev27_fid_gemma_${m}_$t.log; continue; }
  grep -hE "^\[" logs/rev27_fid_gemma_${m}_$t.log | cut -c1-300
done; done
log "사슬 끝"
