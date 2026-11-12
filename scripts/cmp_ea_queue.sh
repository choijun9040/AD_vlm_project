# 개정 15 E-A — DebugUnderflowOverflow 비교, 네 대상 차례로(끝난 것은 건너뜀)
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
O=eval_results/cmp; mkdir -p $O
log(){ echo "[$(date '+%F %T')] $*"; }
run(){ local tag=$1; shift; [ -f $O/ea_$tag.json ] && { log "건너뜀 $tag"; return; }
  log "시작 $tag"; /workspace/venv_main/bin/python scripts/cmp_debug_overflow.py --tag $tag --out $O/ea_$tag.json "$@" > logs/cmp_ea_$tag.log 2>&1
  log "끝 $tag (종료 $?)"; }
run baseline_unfixed --checkpoint checkpoints/student_baseline_v2/epoch_1
run lambda05 --checkpoint checkpoints/sweep_align0.5/step_5000
run resid --checkpoint checkpoints/student_baseline_v2/epoch_1 --vision_state checkpoints_fix/baseline_v2_residual_native.pt
run full --checkpoint checkpoints/student_full/epoch_1
log "E-A 전부 끝"
