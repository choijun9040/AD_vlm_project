# 개정 15d — 비교 처방 그래프(원본 해상도, (R)과 같은 내보내기 설정)를 내보내고, (R)·무교정과 함께 TensorRT 강타입 엔진으로 한 번에 잰다
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
G=onnx/cmp; mkdir -p $G eval_results/cmp
log(){ echo "[$(date '+%F %T')] $*"; }
for A in B1 B2 DTo8 DTo4; do
  N=tower_${A}_nativeh
  if ! { [ -f $G/${N}_p2.onnx.data ] && grep -qs "완료: ${N}_p2.onnx " logs/cmp_export_$A.log; }; then
    log "내보내기 $A"
    /workspace/venv_main/bin/python scripts/export_vision_tower_onnx.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --max_pixels 1440000 --min_pixels 3136 \
      --out $G/$N.onnx --split 16 --attn chunked --dtype fp16 --cmp_arm $A > logs/cmp_export_$A.log 2>&1
    grep -E "^\[대조·비교|완료:|Error|Traceback" logs/cmp_export_$A.log | tail -3
  fi
done
R=onnx/fixfam2r/tower_baseline_v2_1440000h_resid
log "TensorRT 측정(6구성)"
/workspace/venv_ort/bin/python scripts/cmp_trt_eval.py \
  --cfg unfixed:$G/tower_unfixed_nativeh_p1.onnx:$G/tower_unfixed_nativeh_p2.onnx \
  --cfg resid:${R}_p1.onnx:${R}_p2.onnx \
  --cfg B1:$G/tower_B1_nativeh_p1.onnx:$G/tower_B1_nativeh_p2.onnx \
  --cfg B2:$G/tower_B2_nativeh_p1.onnx:$G/tower_B2_nativeh_p2.onnx \
  --cfg DTo8:$G/tower_DTo8_nativeh_p1.onnx:$G/tower_DTo8_nativeh_p2.onnx \
  --cfg DTo4:$G/tower_DTo4_nativeh_p1.onnx:$G/tower_DTo4_nativeh_p2.onnx \
  --out eval_results/cmp/trt_cmp_native.json > logs/cmp_trt_eval.log 2>&1
log "끝 (종료 $?)"; tail -1 logs/cmp_trt_eval.log | cut -c1-300
