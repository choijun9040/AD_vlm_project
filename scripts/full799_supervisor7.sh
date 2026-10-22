#!/bin/bash
# 평가 데이터 전부로 통일 — 감시: 1분마다 각 단계가 살아 있는지 보고, 끝나지 않았는데 죽었으면 다시 띄운다.
# supervisord(/etc/supervisor/conf.d/full799.conf)가 이 스크립트를 띄우므로 컨테이너가 다시 켜져도 이어진다.
cd /workspace/vlm_project_2; O=eval_results/full799; mkdir -p $O logs
log(){ echo "[$(date '+%F %T')] $*"; }
STAGES="queue s4q"
while true; do
  all=1
  for s in $STAGES; do
    [ -f $O/.stage_${s}_done ] && continue
    all=0
    script=scripts/run_full799_${s}.sh; [ $s = queue ] && script=scripts/full799_queue6.sh
    if ! pgrep -f "$script" >/dev/null; then
      log "띄움 $s"; setsid nohup bash $script >> logs/full799_${s}.log 2>&1 < /dev/null &
    fi
  done
  [ $all -eq 1 ] && { log "모든 단계 완료 — 감시 종료"; exit 0; }
  sleep 60
done
