# 개정 13 대기열 교체(2026-11-04): 컨테이너 CPU 할당량이 15.36개분이라(cgroup cpu.max) 프로세스마다 스레드를 제한해 띄운다.
# 지금 도는 측정(백본 첫 묶음·ONNX 첫 그래프)이 끝나기를 기다렸다가 같은 대기열을 이어서 돌린다 — 끝난 것은 건너뛴다.
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
log(){ echo "[$(date '+%F %T')] $*"; }
waitpids(){ for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 60; done; done; }
( waitpids 739460 740296; log "백본 첫 묶음 끝 — 대기열 이어서"; bash scripts/verify799_backbone.sh ) > logs/verify799_backbone_queue2.log 2>&1 &
( waitpids 739543; log "ONNX 첫 그래프 끝 — 대기열 이어서"; bash scripts/verify799_onnx.sh ) > logs/verify799_onnx_trt_queue2.log 2>&1 &
wait
