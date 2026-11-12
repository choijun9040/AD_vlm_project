# 개정 13 ② — 언어 모델 백본 여유를 11,309문항으로 (이미지 전처리·비전 출력 재사용 --cache_images, 앞 40문항에서 원래 경로와 같음 확인)
# 원본 해상도 순전파가 프로세스마다 약 21.7 GB라 2모델씩 돌린다. 끝난 모델은 건너뛴다 — 다시 실행하면 이어서 한다.
cd /workspace/vlm_project_2
O=eval_results/verify799; mkdir -p $O
log(){ echo "[$(date '+%F %T')] $*"; }
for PAIR in "pretrained:none baseline_v2:checkpoints/student_baseline_v2/epoch_1" \
            "full:checkpoints/student_full/epoch_1 spatial:checkpoints/student_spatial/epoch_1" \
            "temporal:checkpoints/student_temporal/epoch_1 kd_only_v4:checkpoints/student_kd_only_v4/epoch_1"; do
  for TC in $PAIR; do
    T=${TC%%:*}; C=${TC#*:}
    [ -f $O/llm_headroom_${T}_q11309.json ] && { log "건너뜀 $T"; continue; }
    log "시작 $T"
    /workspace/venv_main/bin/python scripts/probe_llm_headroom.py --checkpoint $C --tag ${T}_q11309 --all_questions --cache_images \
      --out_dir $O --per_image $O/llm_headroom_${T}_q11309_per_image.jsonl > logs/verify799_backbone_$T.log 2>&1 < /dev/null &
    sleep 30
  done
  wait
  for TC in $PAIR; do T=${TC%%:*}; log "끝 $T — $(grep -E '^\[크기|^\[상실' logs/verify799_backbone_$T.log | tr '\n' ' ')"; done
done
log "백본 전부 끝"
