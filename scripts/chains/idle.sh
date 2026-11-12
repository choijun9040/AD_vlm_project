#!/bin/bash
# 개정 30 B — GPU 전용(지연 4선택지·재빌드 변동·래퍼 fp32 검증), DriveLM 799 (이어 하기 판). 다른 사슬이 다 끝나고 GPU가 빈 뒤, 측정 내내 5초마다 GPU 프로세스 기록.
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/workspace/venv_main/bin/python; PT=/workspace/venv_ort/bin/python
R=/dev/shm/dlm799; L=$R/lat; G=$R/onnx; mkdir -p $L/plans $R/verify
for f in logs/rev30a_qwen.done logs/rev30a_gemma.done logs/rev30b.done; do until [ -f $f ]; do sleep 120; done; done
[ -f $R/.alive_idle ] || { rm -f logs/steps/i_prep.done logs/steps/i_f32graph.done logs/steps/i_densegraph.done; touch $R/.alive_idle; }
IMG=$(head -1 $R/paths799.txt)
idle(){ until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do sleep 60; done; }
step i_prep -- bash -c "$PY scripts/prep_trt_inputs.py --list $R/paths799.txt --limit 799 --max_pixels 12845056 --min_pixels 3136 --out $L/inputs799.npz > logs/dlm_prep799.log 2>&1" || exit 1
step i_f32graph -- bash -c "$PY scripts/export_vision_tower_onnx.py --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG --split 16 --attn chunked --dtype fp32 --out $L/tower_official_dlm1596h_f32.onnx > logs/dlm_export_f32.log 2>&1" || exit 1
idle
# 래퍼 검증 그래프: dense fp32는 원본 해상도(7,296 패치)에서 ONNX 추적 중 GPU 80 GB를 넘는다(2026-11-05 두 번 실패) → 배포 그래프와 같은 조각난 어텐션 fp32(분할 없음)로
step i_densegraph -- bash -c "$PY scripts/export_vision_tower_onnx.py --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG --attn chunked --dtype fp32 --out $R/verify/tower_official_dlm1596h_fp32.onnx >> logs/dlm_export_dense.log 2>&1" || exit 1
idle; log "GPU 비었음 — 감시 시작"
( while :; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 5; done ) >> logs/dlm_gpu_watch.log 2>&1 &
W=$!; trap "kill $W 2>/dev/null" EXIT
check(){ local n=$(awk -v s="$2" -v e="$3" '$1>=s && $1<=e' logs/dlm_gpu_watch.log | awk -F';' '{c=0; for(i=1;i<=NF;i++){split($i,a,", "); gsub(/ MiB/,"",a[2]); if(a[2]+0>100) c++} if(c>1) print}' | wc -l); echo "$1: 겹친 순간 $n" | tee -a logs/dlm_overlap.txt; [ "$n" = 0 ]; }
P="--npz $L/inputs799.npz --plan_dir $L/plans --per_image"
F32="--onnx $L/tower_official_dlm1596h_f32_p1.onnx --onnx2 $L/tower_official_dlm1596h_f32_p2.onnx"
lat(){ local tag=$1; shift; local s=$(date +%T); "$@" > logs/dlm_lat_$tag.log 2>&1 && check dlm_lat_$tag $s $(date +%T); }
step i_lat_f32 -- lat f32 $PT scripts/trt_precision_probe.py $F32 --modes fp32 --save_out $L/ref_fp32.npy --save_mode fp32 $P eval_results/cmp/dlm_lat_f32_per_image.jsonl --out eval_results/cmp/dlm_lat_f32.json
step i_lat_bf16 -- lat bf16 $PT scripts/trt_precision_probe.py $F32 --modes bf16 --cos_ref $L/ref_fp32.npy $P eval_results/cmp/dlm_lat_bf16_per_image.jsonl --out eval_results/cmp/dlm_lat_bf16.json
step i_lat_f16 -- lat f16 $PT scripts/trt_precision_probe.py --onnx $G/tower_official_dlm1596h_p1.onnx --onnx2 $G/tower_official_dlm1596h_p2.onnx --modes typed --cos_ref $L/ref_fp32.npy $P eval_results/cmp/dlm_lat_f16_per_image.jsonl --out eval_results/cmp/dlm_lat_f16.json
step i_lat_f16_resid -- lat f16_resid $PT scripts/trt_precision_probe.py --onnx $G/tower_official_dlm1596h_resid_p1.onnx --onnx2 $G/tower_official_dlm1596h_resid_p2.onnx --modes typed --cos_ref $L/ref_fp32.npy $P eval_results/cmp/dlm_lat_f16_resid_per_image.jsonl --out eval_results/cmp/dlm_lat_f16_resid.json
s=$(date +%T)
rb(){ local s=$(date +%T); RDIR=$R GDIR=onnx GBASE=tower_official_dlm1596h NPZ=$L/inputs799.npz A_PLANS=plans OUT=eval_results/cmp/dlm_rebuild.json $PT scripts/rev28_rebuild.py > logs/dlm_rebuild.log 2>&1 && check dlm_rebuild $s $(date +%T); }
step i_rebuild -- rb
step i_verify -- bash -c "$PT scripts/onnx_verify799.py --runtime trt --attn default --checkpoint none --max_pixels 12845056 --min_pixels 3136 --list $R/paths799.txt --limit 799 --onnx $R/verify/tower_official_dlm1596h_fp32.onnx --plan $R/verify/tower_official_dlm1596h_fp32.plan --out eval_results/cmp/dlm_wrapper_verify.json >> logs/dlm_verify.log 2>&1"
miss=$(for n in i_lat_f32 i_lat_bf16 i_lat_f16 i_lat_f16_resid i_rebuild i_verify; do [ -f logs/steps/$n.done ] || echo $n; done)
[ -z "$miss" ] && { touch logs/dlm_idle.done; log "사슬 끝"; } || { log "**남은 단계: $miss — 다시 걸면 이어서 한다**"; exit 1; }
