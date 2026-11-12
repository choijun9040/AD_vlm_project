#!/bin/bash
# 개정 30 B — 공식 Qwen DriveLM 799: 목록 → c(개정 30 A의 train 100 값) → dlm_orin_chain(가중치·ONNX·A100 엔진 기준·Orin 꾸러미). 2026-11-04
set -u
cd /workspace/vlm_project_2
R=/dev/shm/dlm799; mkdir -p $R
python3 -c "
import json; rows=[json.loads(l) for l in open('data/drivelm_official/val799.jsonl')]
open('$R/paths799.txt','w').write('\n'.join(r['image'] for r in rows)+'\n')
open('$R/img799.txt','w').write('\n'.join(r['image'].split('/')[-1] for r in rows)+'\n')"
until python3 -c "import json,sys; d=json.load(open('eval_results/cmp/dlm_gen_qwen.json')); sys.exit(0 if 'c' in d.get('calib',{}) else 1)" 2>/dev/null; do sleep 60; done
C=$(python3 -c "import json; print(json.load(open('eval_results/cmp/dlm_gen_qwen.json'))['calib']['c'])")
echo "[$(date '+%F %T')] c = $C (train 100, 공식 설정)"
until [ $(( 81920 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) )) -gt 30000 ]; do sleep 60; done
C=$C bash scripts/dlm_orin_chain.sh
