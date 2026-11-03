#!/bin/bash
# 사전 등록 개정 7 — 실험 1(교정이 출력을 바꾸는가)과 실험 3(TensorRT). 원격 A100.
#   nohup bash scripts/fixfam2_eval_trt.sh > logs/fixfam2_eval_trt.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam2; mkdir -p $O
NAT="--max_pixels 1440000 --min_pixels 3136"
RESID=checkpoints_fix/baseline_v2_residual_native.pt
BASE=checkpoints/student_baseline_v2/epoch_1
PT=/workspace/venv_trt/bin/python

# ── 실험 1: (R) 교정 bf16 평가 (11,309문항, 8조각, 4개씩) ─────────────────────────
shard(){
  local k=$1
  step "실험1 (R) bf16 조각 $k/8" $O/student_baseline_v2_epoch_1_native_bf16_resid_shard${k}of8_nuscenesqa_val.json 20 1 \
    $PY scripts/eval_nuscenesqa_val.py --checkpoint $BASE --dump_predictions --log_every 1000 --out_dir $O $NAT \
    --lora_dtype bfloat16 --vision_state $RESID --tag native_bf16_resid --shard $k --num_shards 8
}
for grp in "0 1 2 3" "4 5 6 7"; do
  for k in $grp; do shard $k > logs/fixfam2_exp1_shard$k.log 2>&1 & sleep 90; done
  wait
done
grep -h "시작\|끝\|실패\|포기" logs/fixfam2_exp1_shard*.log
$PY scripts/compare_fix_predictions.py --fix "$O/student_baseline_v2_epoch_1_native_bf16_resid_shard*of8_predictions.jsonl" \
  --ref "eval_results/full799/student_baseline_v2_epoch_1_native_bf16_shard*of8_predictions.jsonl" --out $O/exp1_A_resid_bf16_vs_bf16.json
$PY scripts/compare_fix_predictions.py --fix "eval_results/fixfam/student_baseline_v2_epoch_1_native_fp16_resid_shard*of8_predictions.jsonl" \
  --ref "$O/student_baseline_v2_epoch_1_native_bf16_resid_shard*of8_predictions.jsonl" --out $O/exp1_B_resid_fp16_vs_resid_bf16.json
stage_done fixfam2_exp1

# ── 실험 3: TensorRT — (R) 교정 타워, 원본 해상도, 799장을 두 번(400·399)에 나눠 잰다(디스크) ──────────
G=onnx/fixfam2; mkdir -p $G
HARGS="--split 16 --attn chunked --dtype fp16"
exp(){ local name=$1; shift
  if [ -f $G/$name.onnx ] || [ -f $G/${name}_p2.onnx.data ]; then log "  [건너뜀] 그래프 $name 있음"; return 0; fi
  wait_disk 8
  log "시작 내보내기 $name"; $PY scripts/export_vision_tower_onnx.py --checkpoint $BASE $NAT --out $G/$name.onnx "$@" > logs/fixfam2_export_$name.log 2>&1
  local rc=$?; grep -E "^\[|일치|통과|Error|Traceback" logs/fixfam2_export_$name.log | tail -6
  [ $rc -eq 0 ] && log "끝 내보내기 $name" || { log "실패 내보내기 $name (종료 $rc)"; return 1; }
}
probe(){ local name=$1 npz=$2 modes=$3 g=$4 split=$5; shift 5
  local out=$O/$name.json; [ -f $out ] && { log "  [건너뜀] $out"; return 0; }
  local a=(--onnx $G/$g.onnx); [ "$split" = split ] && a=(--onnx $G/${g}_p1.onnx --onnx2 $G/${g}_p2.onnx)
  wait_gpu 30; log "시작 $name ($modes)"
  $PT scripts/trt_precision_probe.py "${a[@]}" --npz $npz --modes $modes --out $out --per_image $O/${name}_per_image.jsonl \
      --plan_dir $G/plans --timing_cache $G/timing.cache "$@" > logs/fixfam2_$name.log 2>&1
  local rc=$?; grep -E "상실|코사인|저장|Error|Traceback" logs/fixfam2_$name.log | tail -8; log "끝 $name (종료 $rc)"
}
exp tower_unfixed_native
exp tower_resid_native --vision_state $RESID
exp tower_resid_nativeh --vision_state $RESID $HARGS
for half in 0 1; do
  off=$((half*400)); n=$([ $half = 0 ] && echo 400 || echo 399)
  NPZ=$G/inputs_native_h$half.npz; REF=$G/ref_unfixed_fp32_h$half.npy
  if [ ! -f $NPZ ]; then wait_disk 20; log "시작 입력 준비 $NPZ ($off부터 $n장)"
    $PY scripts/prep_trt_inputs.py --limit $n --offset $off $NAT --out $NPZ.tmp.npz > logs/fixfam2_prep_h$half.log 2>&1 && mv $NPZ.tmp.npz $NPZ
    log "끝 입력 준비 $NPZ"; fi
  probe exp3_ref_unfixed_fp32_h$half $NPZ fp32 tower_unfixed_native single --save_out $REF --save_mode fp32
  probe exp3_resid_single_h$half $NPZ fp32,fp16 tower_resid_native single --cos_ref $REF
  probe exp3_resid_typed_h$half $NPZ typed tower_resid_nativeh split --cos_ref $REF
  probe exp3_resid_fp16graph_h$half $NPZ fp16 tower_resid_nativeh split --cos_ref $REF
  rm -f $NPZ $REF; log "입력·기준 삭제 h$half (디스크 여유 $(disk_free_gb)GB)"
done
rm -rf $G/*.onnx $G/*.onnx.data $G/plans $G/timing.cache; log "그래프·엔진 삭제 (디스크 여유 $(disk_free_gb)GB)"
stage_done fixfam2_exp3
stage_done fixfam2_evaltrt
