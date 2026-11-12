# 개정 14b — Draw Things 원형(DT-원형) 799장: Qwen DTo8·DTo4, Gemma 4 네 조건 × DTo8·DTo4. 끝난 팔은 건너뛴다(다시 실행하면 이어서).
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
O=eval_results/fixfam2
log(){ echo "[$(date '+%F %T')] $*"; }
log "Qwen DT-원형"
/workspace/venv_main/bin/python scripts/fixed_mult_arms.py qwen --checkpoint checkpoints/student_baseline_v2/epoch_1 --max_pixels 1440000 --min_pixels 3136 \
  --arms DTo8,DTo4 --out $O/exp5_qwen_dto.json > logs/exp5_dto_qwen.log 2>&1
grep -E "^\[DTo" logs/exp5_dto_qwen.log
for M in 26B-A4B 31B; do for S in 280 1120; do
  log "Gemma $M $S DT-원형"
  /workspace/venv_gemma/bin/python scripts/fixed_mult_arms.py gemma --model_dir /workspace/gemma4_vision/gemma-4-${M}-it --max_soft_tokens $S \
    --arms DTo8,DTo4 --out $O/exp5_gemma_dto_${M}_st${S}.json > logs/exp5_dto_gemma_${M}_st${S}.log 2>&1
  grep -E "^\[DTo" logs/exp5_dto_gemma_${M}_st${S}.log
done; done
log "DT-원형 전부 끝"
