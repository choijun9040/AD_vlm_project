#!/bin/bash
# 로컬 감시기 — vast2(새 인스턴스)에 3분마다 접속해 supervise_rev35.sh를 부른다. 접속 실패·회복을 기록. 두 사슬(rev35·sdxl_env2)이 모두 끝나면 스스로 멈춘다.
# 멈추기: touch logs/watchdog_vast2.stop   2026-11-09
cd /home/cj/workspace/vlm_project_2
L=logs/watchdog_vast2.log; fail=0
while [ ! -f logs/watchdog_vast2.stop ]; do
  out=$(ssh -o ConnectTimeout=20 -o BatchMode=yes vast2 'bash /workspace/vlm_project_2/scripts/chains/supervise_rev35.sh' 2>/dev/null | grep -v -i "have fun\|locale\|welcome" | tail -1)
  if [ -z "$out" ]; then fail=$((fail+1)); echo "$(date '+%F %T') **접속 실패 ${fail}회째**" >> $L
  else [ $fail -gt 0 ] && echo "$(date '+%F %T') 접속 회복(실패 $fail회 뒤)" >> $L; fail=0; echo "$out" >> $L
       case "$out" in *실행*|*재시작*|*멈춤*) ;; *끝*) echo "$(date '+%F %T') 사슬 끝 — 감시 종료" >> $L; exit 0;; esac; fi
  sleep 180
done
