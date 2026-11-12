# D* — FLUX 변환기 다운로드 → (GPU 여유 60 GB 이상일 때) 프롬프트 2개 시험 → 본 측정. 2026-11-04
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log(){ echo "[$(date '+%F %T')] $*"; }
log "다운로드 시작"
/workspace/venv_sd/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('black-forest-labs/FLUX.1-schnell', allow_patterns=['model_index.json','scheduler/*','transformer/*','vae/*'])
print(p)"
log "다운로드 끝 (종료 $?)"; df -h /workspace | tail -1
bash scripts/run_when_free.sh 60000 logs/flux_smoke.done logs/flux_smoke.log bash -c \
  "/workspace/venv_sd/bin/python scripts/cmp_flux.py --emb eval_results/cmp/flux_prompts.pt --n 2 --out logs/flux_smoke.json && touch logs/flux_smoke.done"
log "시험 끝"; grep -E "^\[|Error|Traceback" logs/flux_smoke.log | tail -12
bash scripts/run_when_free.sh 60000 eval_results/cmp/flux_cmp.json.done logs/flux_cmp.log bash -c \
  "/workspace/venv_sd/bin/python scripts/cmp_flux.py --emb eval_results/cmp/flux_prompts.pt --out eval_results/cmp/flux_cmp.json && touch eval_results/cmp/flux_cmp.json.done"
log "본 측정 끝"
