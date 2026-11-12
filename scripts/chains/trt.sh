#!/bin/bash
# 개정 30 B — 공식 Qwen DriveLM 799: c → (R) 가중치 → ONNX(896×1596) → A100 엔진 기준 → Orin 꾸러미 (이어 하기 판).
# /dev/shm은 인스턴스를 다시 켜면 사라진다 — 그 경우 그 안의 산출물 단계 표시도 지우고 다시 만든다.
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/workspace/venv_main/bin/python; PT=/workspace/venv_ort/bin/python
R=/dev/shm/dlm799; G=$R/onnx; mkdir -p $G $R/plans
[ -f $R/.alive ] || { rm -f logs/steps/t_state.done logs/steps/t_export_*.done logs/steps/t_pkg.done; touch $R/.alive; }   # 램 디스크가 새로 생겼다
python3 -c "
import json; rows=[json.loads(l) for l in open('data/drivelm_official/val799.jsonl')]
open('$R/paths799.txt','w').write('\n'.join(r['image'] for r in rows)+'\n')
open('$R/img799.txt','w').write('\n'.join(r['image'].split('/')[-1] for r in rows)+'\n')"
until [ -f eval_results/cmp/dlm_gen_qwen.json.calib.json ]; do sleep 60; done
C=$(python3 -c "import json; print(json.load(open('eval_results/cmp/dlm_gen_qwen.json.calib.json'))['c'])"); log "c = $C (train 100)"
until [ $(( 81920 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) )) -gt 30000 ]; do sleep 60; done
IMG=$(head -1 $R/paths799.txt); STATE=$R/official_residual_dlm.pt
HARGS="--split 16 --attn chunked --dtype fp16 --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG"
step t_state -- $PY scripts/make_official_resid_state.py --c $C --out $STATE || exit 1
step t_export_unfixed -- bash -c "$PY scripts/export_vision_tower_onnx.py $HARGS --out $G/tower_official_dlm1596h.onnx > logs/dlm_export_unfixed.log 2>&1" || exit 1
step t_export_resid -- bash -c "$PY scripts/export_vision_tower_onnx.py $HARGS --vision_state $STATE --out $G/tower_official_dlm1596h_resid.onnx > logs/dlm_export_resid.log 2>&1" || exit 1
step t_a100ref eval_results/cmp/dlm_trt_a100.json -- bash -c "$PT scripts/cmp_trt_eval.py --attn default --checkpoint none --max_pixels 12845056 --min_pixels 3136 --list $R/paths799.txt --limit 799 --plan_dir $R/plans \
   --cfg official:$G/tower_official_dlm1596h_p1.onnx:$G/tower_official_dlm1596h_p2.onnx --cfg resid:$G/tower_official_dlm1596h_resid_p1.onnx:$G/tower_official_dlm1596h_resid_p2.onnx \
   --out eval_results/cmp/dlm_trt_a100.json >> logs/dlm_trt_a100.log 2>&1" || exit 1
step t_pkg -- bash scripts/make_orin_pkg_dlm.sh || exit 1
touch logs/rev30b.done; log "사슬 끝"
