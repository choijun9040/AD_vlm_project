#!/bin/bash
# 개정 30 A — Gemma 4, DriveLM (이어 하기 판)
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PG=/workspace/venv_gemma/bin/python; O=eval_results/cmp; V=data/drivelm_official/val799.jsonl; C=data/drivelm_official/train100.jsonl
ok=1
for m in 26B-A4B 31B; do for t in 280 1120; do
  D=/workspace/gemma4_vision/gemma-4-$m-it
  step g_diag_${m}_$t $O/dlm_diag_gemma_${m}_$t.json -- bash -c "$PG scripts/diag_gemma_rwqa.py --model_dir $D --tokens $t --q $V --out $O/dlm_diag_gemma_${m}_$t.json >> logs/dlm_diag_gemma_${m}_$t.log 2>&1" || ok=0
  step g_fid_${m}_$t $O/dlm_fid_gemma_${m}_$t.json -- bash -c "$PG scripts/fidelity_rwqa.py gemma --model_dir $D --tokens $t --q $V --calib_q $C --out $O/dlm_fid_gemma_${m}_$t.json >> logs/dlm_fid_gemma_${m}_$t.log 2>&1" || ok=0
done; done
[ $ok = 1 ] && { touch logs/rev30a_gemma.done; log "사슬 끝"; } || { log "**실패한 단계가 있다**"; exit 1; }
