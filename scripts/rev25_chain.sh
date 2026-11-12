#!/bin/bash
# 개정 25 — 공식 모델 배포 선택지 비교(fp32 / fp16 무교정 / fp16+(R) / bf16), RealWorldQA 952×1456 465장, A100에 다른 GPU 작업이 없을 때. 2026-11-04
set -u
cd /workspace/vlm_project_2
PY=/workspace/venv_main/bin/python; PT=/workspace/venv_ort/bin/python
R=/dev/shm/rwqa952; L=$R/lat; G2=$R/onnx2; mkdir -p $L/plans
IMG=$R/images/$(head -1 $R/img465.txt)
log(){ echo "[$(date '+%F %T')] $*"; }
until [ -f logs/rwqa_orin_chain.done ]; do sleep 120; done
log "fp16 그래프 준비됨(개정 23 사슬)"
[ -f $L/inputs465.npz ] || { log "입력 묶음"; $PY scripts/prep_trt_inputs.py --list $R/paths465.txt --limit 465 --max_pixels 12845056 --min_pixels 3136 --out $L/inputs465.npz > logs/rev25_prep.log 2>&1 || { tail -3 logs/rev25_prep.log; exit 1; }; }
[ -f $L/tower_official_rwqa952h_f32_p2.onnx.data ] || { log "fp32 그래프 내보내기"; $PY scripts/export_vision_tower_onnx.py --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG \
    --split 16 --attn chunked --dtype fp32 --out $L/tower_official_rwqa952h_f32.onnx > logs/rev25_export.log 2>&1 || { tail -5 logs/rev25_export.log; exit 1; }; }
# 다른 GPU 작업이 없을 때까지 — 지연 측정
while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 1000 ] && break; sleep 120; done
log "GPU 비었음 (${u} MiB) — 측정 시작"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > logs/rev25_gpu_at_start.txt
P="--npz $L/inputs465.npz --plan_dir $L/plans --per_image"
F32="--onnx $L/tower_official_rwqa952h_f32_p1.onnx --onnx2 $L/tower_official_rwqa952h_f32_p2.onnx"
$PT scripts/trt_precision_probe.py $F32 --modes fp32 --save_out $L/ref_fp32.npy --save_mode fp32 $P eval_results/cmp/rev25_f32_per_image.jsonl \
    --out eval_results/cmp/rev25_f32.json > logs/rev25_f32.log 2>&1 || { tail -5 logs/rev25_f32.log; exit 1; }
$PT scripts/trt_precision_probe.py $F32 --modes bf16 --cos_ref $L/ref_fp32.npy $P eval_results/cmp/rev25_bf16_per_image.jsonl \
    --out eval_results/cmp/rev25_bf16.json > logs/rev25_bf16.log 2>&1 || { tail -5 logs/rev25_bf16.log; exit 1; }
for m in "" _resid; do
  $PT scripts/trt_precision_probe.py --onnx $G2/tower_official_rwqa952h${m}_p1.onnx --onnx2 $G2/tower_official_rwqa952h${m}_p2.onnx --modes typed \
      --cos_ref $L/ref_fp32.npy $P eval_results/cmp/rev25_f16${m}_per_image.jsonl --out eval_results/cmp/rev25_f16${m}.json > logs/rev25_f16${m}.log 2>&1 || { tail -5 logs/rev25_f16${m}.log; exit 1; }
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > logs/rev25_gpu_at_end.txt
grep -hE "상실|지연|가중치|코사인" logs/rev25_f32.log logs/rev25_bf16.log logs/rev25_f16.log logs/rev25_f16_resid.log | cut -c1-200
log "사슬 끝"
