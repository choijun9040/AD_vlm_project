# GPU 여유가 25 GB 이상일 때 FLUX 프롬프트 임베딩을 다시 시작한다(메모리 부족으로 멈춘 뒤, 2026-11-04)
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4
until [ -f eval_results/cmp/flux_prompts.pt ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ $((81920 - used)) -ge 25000 ]; then
    echo "[$(date '+%F %T')] 여유 $((81920 - used)) MiB — 시작"
    /workspace/venv_sd/bin/python scripts/flux_embed.py --tsv data/PartiPrompts.tsv --out eval_results/cmp/flux_prompts.pt >> logs/flux_embed.log 2>&1 < /dev/null
    echo "[$(date '+%F %T')] 종료 $?"
  fi
  sleep 120
done
echo "[$(date '+%F %T')] 임베딩 있음 — 끝"
