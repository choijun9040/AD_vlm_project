#!/bin/bash
# 개정 30 B — DriveLM GPU 전용 사슬: 지연 4선택지(개정 25), 재빌드 변동(개정 28), 래퍼 fp32 검증(개정 24 ①). 개정 30 A·B가 끝나고 GPU가 빈 뒤, 측정 내내 5초마다 GPU 프로세스 기록. 2026-11-04
set -u
cd /workspace/vlm_project_2
PY=/workspace/venv_main/bin/python; PT=/workspace/venv_ort/bin/python
R=/dev/shm/dlm799; L=$R/lat; G=$R/onnx; mkdir -p $L/plans $R/verify
IMG=$(head -1 $R/paths799.txt)
log(){ echo "[$(date '+%F %T')] $*"; }
for f in logs/rev30a_qwen.done logs/rev30a_gemma.done logs/rev30b.done; do until [ -f $f ]; do sleep 120; done; done
log "준비: 입력 묶음·fp32 분할 그래프·fp32 dense 그래프"
[ -f $L/inputs799.npz ] || $PY scripts/prep_trt_inputs.py --list $R/paths799.txt --limit 799 --max_pixels 12845056 --min_pixels 3136 --out $L/inputs799.npz > logs/dlm_prep799.log 2>&1
[ -f $L/tower_official_dlm1596h_f32_p2.onnx.data ] || $PY scripts/export_vision_tower_onnx.py --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG --split 16 --attn chunked --dtype fp32 --out $L/tower_official_dlm1596h_f32.onnx > logs/dlm_export_f32.log 2>&1
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do sleep 60; done
[ -f $R/verify/tower_official_dlm1596h_fp32.onnx ] || $PY scripts/export_vision_tower_onnx.py --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG --out $R/verify/tower_official_dlm1596h_fp32.onnx > logs/dlm_export_dense.log 2>&1
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do sleep 60; done
log "GPU 비었음 — 감시 시작"
( while :; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 5; done ) > logs/dlm_gpu_watch.log 2>&1 &
W=$!
check(){ local n=$(awk -v s="$2" -v e="$3" '$1>=s && $1<=e' logs/dlm_gpu_watch.log | awk -F';' '{c=0; for(i=1;i<=NF;i++){split($i,a,", "); gsub(/ MiB/,"",a[2]); if(a[2]+0>100) c++} if(c>1) print}' | wc -l); echo "$1: 겹친 순간 $n" | tee -a logs/dlm_overlap.txt; }
P="--npz $L/inputs799.npz --plan_dir $L/plans --per_image"
F32="--onnx $L/tower_official_dlm1596h_f32_p1.onnx --onnx2 $L/tower_official_dlm1596h_f32_p2.onnx"
s=$(date +%T); $PT scripts/trt_precision_probe.py $F32 --modes fp32 --save_out $L/ref_fp32.npy --save_mode fp32 $P eval_results/cmp/dlm_lat_f32_per_image.jsonl --out eval_results/cmp/dlm_lat_f32.json > logs/dlm_lat_f32.log 2>&1; check dlm_lat_f32 $s $(date +%T)
s=$(date +%T); $PT scripts/trt_precision_probe.py $F32 --modes bf16 --cos_ref $L/ref_fp32.npy $P eval_results/cmp/dlm_lat_bf16_per_image.jsonl --out eval_results/cmp/dlm_lat_bf16.json > logs/dlm_lat_bf16.log 2>&1; check dlm_lat_bf16 $s $(date +%T)
for m in "" _resid; do
  s=$(date +%T); $PT scripts/trt_precision_probe.py --onnx $G/tower_official_dlm1596h${m}_p1.onnx --onnx2 $G/tower_official_dlm1596h${m}_p2.onnx --modes typed --cos_ref $L/ref_fp32.npy $P eval_results/cmp/dlm_lat_f16${m}_per_image.jsonl --out eval_results/cmp/dlm_lat_f16${m}.json > logs/dlm_lat_f16${m}.log 2>&1; check dlm_lat_f16$m $s $(date +%T)
done
grep -hE "지연|상실|코사인" logs/dlm_lat_*.log | cut -c1-200
s=$(date +%T); log "재빌드 변동"
RDIR=$R GDIR=onnx GBASE=tower_official_dlm1596h NPZ=$L/inputs799.npz A_PLANS=plans OUT=eval_results/cmp/dlm_rebuild.json $PT scripts/rev28_rebuild.py > logs/dlm_rebuild.log 2>&1; check dlm_rebuild $s $(date +%T)
grep -hE "^\[" logs/dlm_rebuild.log
s=$(date +%T); log "래퍼 fp32 검증"
$PT scripts/onnx_verify799.py --runtime trt --attn default --checkpoint none --max_pixels 12845056 --min_pixels 3136 --list $R/paths799.txt --limit 799 \
   --onnx $R/verify/tower_official_dlm1596h_fp32.onnx --plan $R/verify/tower_official_dlm1596h_fp32.plan --out eval_results/cmp/dlm_wrapper_verify.json > logs/dlm_verify.log 2>&1; check dlm_verify $s $(date +%T)
tail -3 logs/dlm_verify.log | cut -c1-300
kill $W; log "사슬 끝"
