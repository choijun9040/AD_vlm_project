#!/bin/bash
# 개정 30 B — 공식 Qwen DriveLM 896×1596(원본 1600×900) 799장: (R) 가중치(c = ¼, 2단계 측정용 100문항) → ONNX 두 모델 → A100 fp16 엔진 기준 → Orin 꾸러미. 산출물은 /dev/shm(디스크 부족)
set -u
cd /workspace/vlm_project_2
PY=/workspace/venv_main/bin/python; PT=/workspace/venv_ort/bin/python
R=/dev/shm/dlm799; G=$R/onnx; mkdir -p $G $R/plans   # onnx/·plans/는 두 사슬이 겹쳐 쓴 판(2026-11-04 21:59) — 쓰지 않는다
IMG=$(head -1 $R/paths799.txt)
HARGS="--split 16 --attn chunked --dtype fp16 --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG"
log(){ echo "[$(date '+%F %T')] $*"; }
STATE=$R/official_residual_dlm.pt
[ -f $STATE ] || { log "(R) 가중치"; $PY scripts/make_official_resid_state.py --c ${C:?c를 넘길 것} --out $STATE || exit 1; }
[ -f $G/tower_official_dlm1596h_p2.onnx.data ] || { log "내보내기 무교정"; $PY scripts/export_vision_tower_onnx.py $HARGS --out $G/tower_official_dlm1596h.onnx > logs/dlm_export_unfixed.log 2>&1 || { tail -5 logs/dlm_export_unfixed.log; exit 1; }; }
[ -f $G/tower_official_dlm1596h_resid_p2.onnx.data ] || { log "내보내기 (R)"; $PY scripts/export_vision_tower_onnx.py $HARGS --vision_state $STATE --out $G/tower_official_dlm1596h_resid.onnx > logs/dlm_export_resid.log 2>&1 || { tail -5 logs/dlm_export_resid.log; exit 1; }; }
grep -hE "고정 형상|일치|통과|오차" logs/dlm_export_unfixed.log logs/dlm_export_resid.log | tail -8
log "A100 fp16 엔진 기준 (799장)"
$PT scripts/cmp_trt_eval.py --attn default --checkpoint none --max_pixels 12845056 --min_pixels 3136 --list $R/paths799.txt --limit 799 --plan_dir $R/plans \
   --cfg official:$G/tower_official_dlm1596h_p1.onnx:$G/tower_official_dlm1596h_p2.onnx \
   --cfg resid:$G/tower_official_dlm1596h_resid_p1.onnx:$G/tower_official_dlm1596h_resid_p2.onnx \
   --out eval_results/cmp/dlm_trt_a100.json > logs/dlm_trt_a100.log 2>&1 || { tail -5 logs/dlm_trt_a100.log; exit 1; }
tail -1 logs/dlm_trt_a100.log | cut -c1-600
log "꾸러미"; bash scripts/make_orin_pkg_dlm.sh
log "사슬 끝"
