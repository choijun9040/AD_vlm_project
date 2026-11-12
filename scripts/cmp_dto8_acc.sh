# 개정 14b 규칙 — DTo8(Qwen, PyTorch) 상실 0/799 → 11,309문항 정확도·bf16 일치율.
# GPU 여유가 14 GB 이상일 때만 조각을 하나씩 띄운다(메모리 부족으로 조각 2·3이 죽은 뒤 고침, 2026-11-04). 8조각이 모두 끝나야 일치율을 낸다.
cd /workspace/vlm_project_2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
O=eval_results/fixfam2; B=checkpoints/student_baseline_v2/epoch_1; T=native_fp16_dto8
log(){ echo "[$(date '+%F %T')] $*"; }
running(){ ps -eo args | awk -v k="--shard $1 " 'index($0,"scripts/eval_nuscenesqa_val.py") && index($0,"dto8") && index($0" ",k)' | grep -c .; }
while true; do
  left=0
  for k in 0 1 2 3 4 5 6 7; do
    [ -f $O/student_baseline_v2_epoch_1_${T}_shard${k}of8_nuscenesqa_val.json ] && continue
    left=$((left+1))
    [ "$(running $k)" != 0 ] && continue
    free=$((81920 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)))
    [ $free -lt 14000 ] && continue
    log "조각 $k 시작 (여유 ${free} MiB)"
    /workspace/venv_main/bin/python scripts/eval_nuscenesqa_val.py --checkpoint $B --dump_predictions --log_every 1000 --out_dir $O \
      --max_pixels 1440000 --min_pixels 3136 --lora_dtype float16 --dt_original 8 --tag $T --shard $k --num_shards 8 \
      > logs/cmp_dto8_acc_$k.log 2>&1 < /dev/null &
    sleep 90
  done
  [ $left = 0 ] && break
  sleep 120
done
/workspace/venv_main/bin/python scripts/compare_fix_predictions.py --fix "$O/student_baseline_v2_epoch_1_${T}_shard*of8_predictions.jsonl" \
  --ref "eval_results/full799/student_baseline_v2_epoch_1_native_bf16_shard*of8_predictions.jsonl" --out $O/cmp_dto8_vs_bf16.json
log "DTo8 정확도 끝"
