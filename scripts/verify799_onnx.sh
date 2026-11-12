# 개정 13 ① — ONNX 래퍼 검증 799장 (그래프 넷: 기준 모델·Full × 200,704·1,440,000, fp32·dense, export 기본값)
# 런타임은 TensorRT fp32 엔진(TF32 끔) — ONNX 런타임 CPU가 며칠 걸려 바꿨다(결과 보기 전, 사용자 「A로 진행」). 다시 실행하면 이어서 한다.
cd /workspace/vlm_project_2
G=onnx/verify799; O=eval_results/verify799; mkdir -p $G/plans $O
log(){ echo "[$(date '+%F %T')] $*"; }
for TC in baseline_v2:checkpoints/student_baseline_v2/epoch_1 full:checkpoints/student_full/epoch_1; do
  T=${TC%%:*}; C=${TC#*:}
  for S in 200704 native; do
    if [ $S = 200704 ]; then N=tower_$T; MP=200704; else N=tower_${T}_native; MP=1440000; fi
    # 그래프 파일이 있어도 내보내기 로그에 「완료」가 없으면 다시 만든다(인스턴스가 쓰는 도중 꺼진 경우)
    if ! { [ -f $G/$N.onnx.data ] && grep -qs "완료: $N.onnx " logs/verify799_export_$N.log logs/verify799_export_baseline_v2_200704.log; }; then
      log "내보내기 $N"; /workspace/venv_main/bin/python scripts/export_vision_tower_onnx.py --checkpoint $C --max_pixels $MP --out $G/$N.onnx > logs/verify799_export_$N.log 2>&1
      grep -E "^\[대조\]|완료|Error|Traceback" logs/verify799_export_$N.log | tail -3
    fi
    [ -f $O/trt_onnx_$N.json ] && { log "건너뜀 $N"; continue; }
    log "검증 $N (TensorRT fp32)"
    /workspace/venv_ort/bin/python scripts/onnx_verify799.py --runtime trt --checkpoint $C --max_pixels $MP --onnx $G/$N.onnx \
      --plan $G/plans/$N.fp32.plan --out $O/trt_onnx_$N.json > logs/verify799_trt_$N.log 2>&1 < /dev/null
    log "끝 $N (종료 $?)"
  done
done
log "ONNX 검증 전부 끝"
