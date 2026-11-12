#!/bin/bash
# 로컬 감시기 — 3분마다 A100 접속을 확인하고 서버 감시(scripts/chains/supervise.sh)를 부른다. 이 대화와 무관하게 돈다(setsid nohup). 2026-11-05
# 멈추기: kill $(cat logs/watchdog_local.pid)
cd /home/cj/workspace/vlm_project_2
echo $$ > logs/watchdog_local.pid
fail=0
while :; do
  out=$(timeout 60 ssh -o ConnectTimeout=20 -o BatchMode=yes vast 'bash /workspace/vlm_project_2/scripts/chains/supervise.sh' 2>/dev/null | grep -E "^20[0-9]{2}-" | tail -1)
  if [ -n "$out" ]; then
    [ $fail -gt 0 ] && echo "$(date '+%F %T') 접속 회복 (실패 $fail회 뒤)"
    fail=0; echo "$out"
  else
    fail=$((fail+1)); echo "$(date '+%F %T') **접속 실패 $fail회째**"
  fi
  [ -f logs/watchdog_local.stop ] && { echo "$(date '+%F %T') 멈춤 파일 — 끝"; exit 0; }
  sleep 180
done
