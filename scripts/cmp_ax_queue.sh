# 개정 16 A* — DebugUnderflowOverflow 문서 그대로(모델 전체, 처음 감지에서 멈춤), 네 대상 차례로
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
O=eval_results/cmp; mkdir -p $O
log(){ echo "[$(date '+%F %T')] $*"; }
run(){ local tag=$1; shift; [ -f $O/ax_$tag.json ] && { log "건너뜀 $tag"; return; }
  log "시작 $tag"; /workspace/venv_main/bin/python scripts/cmp_debug_overflow_exact.py --tag $tag --out $O/ax_$tag.json "$@" > logs/cmp_ax_$tag.log 2>&1 < /dev/null
  log "끝 $tag (종료 $?) $(grep -E "처음 감지|경고 없음" logs/cmp_ax_$tag.log | tail -1)"; }
run baseline_unfixed --checkpoint checkpoints/student_baseline_v2/epoch_1
run lambda05 --checkpoint checkpoints/sweep_align0.5/step_5000
run resid --checkpoint checkpoints/student_baseline_v2/epoch_1 --vision_state checkpoints_fix/baseline_v2_residual_native.pt
run full --checkpoint checkpoints/student_full/epoch_1
log "A* 전부 끝"
