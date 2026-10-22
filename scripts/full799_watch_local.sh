#!/bin/bash
# 로컬에서 원격 A100을 지켜본다 — 끊겼다가 돌아오면 측정 대기열을 6판으로 맞추고(full799_redeploy.sh와 같은 절차) 이어 간다.
# 2026-10-22: 6판(감시 full799g)으로 바꿨다. 5판 감시 full799f는 「내릴 옛 감시」다 — 이 점검이 full799f를 되살리면 대기열이 둘이 된다.
# 접속이 유지되는 동안에도 10분마다 감시 full799g가 도는지, GPU 오류로 생긴 포기 표시가 없는지 확인한다.
# 로컬 백그라운드 작업의 시간 한도(2시간) 안에서 끝나도록 DURATION초(기본 6900) 뒤 요약을 남기고 끝난다 — 다시 걸어 쓴다.
#   HOST=<주소> PORT=<포트> bash scripts/full799_watch_local.sh
R="ssh -o BatchMode=yes -o ConnectTimeout=10 -p ${PORT:?PORT를 주세요} root@${HOST:?HOST를 주세요}"
END=$(( $(date +%s) + ${DURATION:-6900} ))
FIX='cd /workspace/vlm_project_2
for c in full799d full799e full799f; do [ -f /etc/supervisor/conf.d/$c.conf ] && { supervisorctl stop $c >/dev/null 2>&1; rm -f /etc/supervisor/conf.d/$c.conf; echo "옛 감시 해제 $c"; }; done
O=eval_results/full799
for f in $O/.failed_*; do [ -f "$f" ] || continue; k=${f#$O/.failed_}; L=logs/full799_job_${k%.*}.log; if grep -qE "Error 802|cudaGetDeviceCount|system not yet initialized" $L 2>/dev/null; then rm -f $f $O/.attempts_$k; echo "GPU 오류 포기 표시 지움: $k"; fi; done
[ -f /etc/supervisor/conf.d/full799g.conf ] || { git pull -q; cp scripts/full799_supervisor7.conf /etc/supervisor/conf.d/full799g.conf; echo "감시 full799g 등록"; }
supervisorctl reread >/dev/null; supervisorctl update >/dev/null
st=$(supervisorctl status full799g | awk "{print \$2}"); [ "$st" = RUNNING ] || { supervisorctl start full799g >/dev/null; echo "감시 full799g 시작 (이전 상태 $st)"; }
echo "상태 $(date -u +%T)UTC 컨테이너시작 $(ps -o lstart= -p 1 | awk "{print \$4}") 완료 $(ls $O/.done_* 2>/dev/null | wc -l)/168 포기 $(ls $O/.failed_* 2>/dev/null | wc -l) GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>&1 | head -1)"'
down=0; last=0; ndown=0
log(){ echo "[$(date '+%F %T KST')] $*"; }
log "감시 시작"
while [ $(date +%s) -lt $END ]; do
  if timeout 25 $R true >/dev/null 2>&1; then
    if [ $down = 1 ]; then log "접속 복구 — 교체 절차"; sleep 20; timeout 300 $R "$FIX" 2>&1 | grep -v "Welcome\|Have fun\|setlocale"; down=0; last=$(date +%s)
    elif [ $(( $(date +%s) - last )) -ge 600 ]; then timeout 120 $R "$FIX" 2>&1 | grep -v "Welcome\|Have fun\|setlocale" | sed "s/^/  /"; last=$(date +%s); fi
  else
    [ $down = 0 ] && { log "접속 끊김"; ndown=$((ndown+1)); }; down=1
  fi
  sleep 60
done
log "감시 종료(시간 한도) — 끊김 ${ndown}회, 지금 $( [ $down = 1 ] && echo 끊김 || echo 접속 )"
