#!/bin/bash
# 컨테이너 자원 기록 — 30초마다 스레드 수·메모리·GPU 메모리를 남긴다 (2026-10-21, 컨테이너 재시작 두 번 뒤).
# 재시작하면 cgroup 기록이 초기화되므로, 직전 값을 디스크에 남겨 원인을 가린다.
L=/workspace/vlm_project_2/logs/cgroup_watch.log
while true; do
  echo "$(date '+%F %T') pids $(cat /sys/fs/cgroup/pids.current)/$(cat /sys/fs/cgroup/pids.max) mem $(( $(cat /sys/fs/cgroup/memory.current)/1073741824 ))GB gpu $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null) pids_max_ev $(awk '/max/{print $2}' /sys/fs/cgroup/pids.events)" >> $L
  sleep 30
done
