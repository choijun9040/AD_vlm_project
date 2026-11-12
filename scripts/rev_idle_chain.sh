#!/bin/bash
# GPU 전용 사슬 — 개정 25 지연 재측정 + 개정 28 재빌드. 다른 A100 작업이 모두 끝난 뒤, 측정 내내 GPU 프로세스를 5초마다 기록한다. 2026-11-04
set -u
cd /workspace/vlm_project_2
PT=/workspace/venv_ort/bin/python
R=/dev/shm/rwqa952; L=$R/lat; G2=$R/onnx2
log(){ echo "[$(date '+%F %T')] $*"; }
for f in logs/rwqa_eval3.done logs/rev27.done logs/rev24.done logs/rev25.done; do until [ -f $f ]; do sleep 120; done; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 1000 ]; do sleep 60; done
log "다른 작업 없음 — 감시 시작"
( while :; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 5; done ) > logs/idle_gpu_watch.log 2>&1 &
W=$!
check(){  # $1 = 구간 이름 — 그 구간 동안 100 MiB 넘는 GPU 프로세스가 둘 이상인 순간이 있었는가
  local n=$(awk -v s="$2" -v e="$3" '$1>=s && $1<=e' logs/idle_gpu_watch.log | awk -F';' '{c=0; for(i=1;i<=NF;i++){split($i,a,", "); gsub(/ MiB/,"",a[2]); if(a[2]+0>100) c++} if(c>1) print}' | wc -l)
  echo "$1: 겹친 순간 $n" | tee -a logs/idle_overlap.txt
}
P="--npz $L/inputs465.npz --plan_dir $L/plans --per_image"
F32="--onnx $L/tower_official_rwqa952h_f32_p1.onnx --onnx2 $L/tower_official_rwqa952h_f32_p2.onnx"
for spec in "f32|$F32|fp32" "bf16|$F32|bf16" "f16|--onnx $G2/tower_official_rwqa952h_p1.onnx --onnx2 $G2/tower_official_rwqa952h_p2.onnx|typed" "f16_resid|--onnx $G2/tower_official_rwqa952h_resid_p1.onnx --onnx2 $G2/tower_official_rwqa952h_resid_p2.onnx|typed"; do
  IFS='|' read -r tag g mode <<< "$spec"
  s=$(date +%T); log "지연 재측정 $tag"
  $PT scripts/trt_precision_probe.py $g --modes $mode $P eval_results/cmp/rev25b_${tag}_per_image.jsonl --out eval_results/cmp/rev25b_${tag}.json > logs/rev25b_${tag}.log 2>&1
  check "rev25b_$tag" $s $(date +%T); grep -hE "지연|상실" logs/rev25b_${tag}.log | head -3
done
s=$(date +%T); log "개정 28 재빌드"
$PT scripts/rev28_rebuild.py > logs/rev28.log 2>&1; check "rev28" $s $(date +%T)
grep -hE "^\[" logs/rev28.log
kill $W
log "사슬 끝"
