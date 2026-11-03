#!/bin/bash
# 사전 등록 개정 10 — (R) 잔차 재조정의 TensorRT(A100) 검증을 Orin과 같은 해상도(802,816·401,408)에서, 그리고 Orin 꾸러미 만들기.
# 실험 3(원본 해상도)이 끝난 뒤 돈다(디스크·GPU를 나눠 쓰지 않게). 다시 실행하면 끝난 내보내기·측정은 건너뛴다.
#   nohup bash scripts/fixfam2_rtrt.sh > logs/fixfam2_rtrt.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam2; G=onnx/fixfam2r; mkdir -p $O $G
RESID=checkpoints_fix/baseline_v2_residual_native.pt
BASE=checkpoints/student_baseline_v2/epoch_1
PT=/workspace/venv_trt/bin/python
HARGS="--split 16 --attn chunked --dtype fp16"
log "대기: 실험 3(원본 해상도) 완료 표시"
until [ -f $O/.stage_fixfam2_exp3_done ]; do sleep 300; done
exp(){ local name=$1 mp=$2; shift 2
  if [ -f $G/$name.onnx ] || [ -f $G/${name}_p2.onnx.data ]; then log "  [건너뜀] 그래프 $name 있음"; return 0; fi
  wait_disk 8; log "시작 내보내기 $name"
  $PY scripts/export_vision_tower_onnx.py --checkpoint $BASE --max_pixels $mp --min_pixels 3136 --out $G/$name.onnx "$@" > logs/fixfam2r_export_$name.log 2>&1
  local rc=$?; grep -E "^\[|일치|통과|Error|Traceback" logs/fixfam2r_export_$name.log | tail -6
  [ $rc -eq 0 ] && log "끝 내보내기 $name" || { log "실패 내보내기 $name (종료 $rc)"; return 1; }
}
probe(){ local name=$1 npz=$2 modes=$3 g=$4 split=$5; shift 5
  local out=$O/$name.json; [ -f $out ] && { log "  [건너뜀] $out"; return 0; }
  local a=(--onnx $G/$g.onnx); [ "$split" = split ] && a=(--onnx $G/${g}_p1.onnx --onnx2 $G/${g}_p2.onnx)
  wait_gpu 30; log "시작 $name ($modes)"
  $PT scripts/trt_precision_probe.py "${a[@]}" --npz $npz --modes $modes --out $out --per_image $O/${name}_per_image.jsonl \
      --plan_dir $G/plans --timing_cache $G/timing.cache "$@" > logs/fixfam2r_$name.log 2>&1
  local rc=$?; grep -E "상실|코사인|저장|Error|Traceback" logs/fixfam2r_$name.log | tail -8; log "끝 $name (종료 $rc)"
}
for MP in 802816 401408; do
  R=${MP}h
  exp tower_unfixed_$MP $MP
  exp tower_baseline_v2_${R}_resid $MP --vision_state $RESID $HARGS      # Orin 꾸러미에 들어가는 이름
  for half in 0 1; do
    off=$((half*400)); n=$([ $half = 0 ] && echo 400 || echo 399)
    NPZ=$G/inputs_${MP}_h$half.npz; REF=$G/ref_unfixed_fp32_${MP}_h$half.npy
    if [ ! -f $NPZ ]; then wait_disk 12; log "시작 입력 준비 $NPZ ($off부터 $n장)"
      $PY scripts/prep_trt_inputs.py --limit $n --offset $off --max_pixels $MP --min_pixels 3136 --out $NPZ.tmp.npz > logs/fixfam2r_prep_${MP}_h$half.log 2>&1 && mv $NPZ.tmp.npz $NPZ
      log "끝 입력 준비 $NPZ"; fi
    probe rtrt_ref_unfixed_fp32_${MP}_h$half $NPZ fp32 tower_unfixed_$MP single --save_out $REF --save_mode fp32
    # 관행 --fp16 측정은 뺐다 — 배포 경로를 강타입으로 한정(2026-11-03 사용자 결정 「안 A」)
    probe rtrt_resid_typed_${MP}_h$half $NPZ typed tower_baseline_v2_${R}_resid split --cos_ref $REF
    rm -f $NPZ $REF; log "입력·기준 삭제 ${MP} h$half (디스크 여유 $(disk_free_gb)GB)"
  done
  rm -f $G/tower_unfixed_$MP.onnx*        # 분할 fp16 그래프(*h_resid)는 Orin 꾸러미용으로 남긴다
done
rm -rf $G/plans $G/timing.cache
bash scripts/make_orin_pkg_resid.sh && log "Orin 꾸러미 완료" || log "**Orin 꾸러미 실패**"
stage_done fixfam2_rtrt
