#!/bin/bash
# 평가 데이터 전부로 통일 — 작업 대기열 (사전 등록 4fddeb4)
# GPU SM 활동도가 12~30%뿐이라(이미지·문항을 하나씩 처리해 CPU↔GPU 왕복에 묶인다) 독립 작업을 여러 개 함께 돌린다.
# 작업자 K개가 full799_jobs.sh의 목록에서 하나씩 가져간다. 완료 표시·재시도·잠금·자원 대기를 둔다.
#   bash scripts/full799_queue.sh [K]     # 기본 5
set -u; source /workspace/vlm_project_2/scripts/full799_lib.sh
K=${1:-5}
JOBS=$O/.jobs.txt; bash scripts/full799_jobs.sh > $JOBS
# 오케스트레이터 교체 전에 시작된 실행 — 명령줄 패턴으로 알아본다(결과 파일 이름이 명령줄에 없는 경우가 있다)
running_pat(){ case "$1" in
  *_nuscenesqa_val.json) local b; b=$(basename "$1"); b=${b%_nuscenesqa_val.json}; local m=${b%%_epoch_1_*}; local t=${b#*_epoch_1_}; echo "checkpoints/$m/epoch_1 .*--tag $t\$";;
  *) local k=$(basename "$1"); echo "${k%.*}";; esac; }
cleanup_locks(){ for d in $O/.lock_* $O/.grp_*; do [ -d "$d" ] || continue
  p=$(cat "$d/pid" 2>/dev/null); if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then rm -rf "$d"; log "낡은 잠금 정리 $(basename $d)"; fi; done; }
worker(){
  local w=$1
  while true; do
    local remaining=0 ran=0
    while IFS='|' read -r name out gpu disk grp cmd; do
      local key; key=$(basename "$out"); local done=$O/.done_$key
      [ -f "$done" ] && continue
      [ -f "$O/.failed_$key" ] && continue
      remaining=1
      local pat; pat=$(running_pat "$out")
      if [ ! -d "$O/.lock_$key" ] && pgrep -f -- "$pat" >/dev/null; then continue; fi          # 앞선 실행이 아직 돈다
      if [ ! -d "$O/.lock_$key" ] && [ -s "$out" ] && [ -f "$O/.adopt_$key" ]; then              # 앞선 실행이 끝났다
        touch "$done"; log "[w$w] 완료로 인정 $name (앞선 실행의 결과)"; continue; fi
      mkdir "$O/.lock_$key" 2>/dev/null || continue                                              # 다른 작업자가 가져갔다
      echo $$ > "$O/.lock_$key/pid"
      if [ "$grp" != gpu ]; then
        if ! mkdir "$O/.grp_$grp" 2>/dev/null; then rm -rf "$O/.lock_$key"; continue; fi
        echo $$ > "$O/.grp_$grp/pid"; fi
      ran=1
      local n; n=$(( $(cat "$O/.attempts_$key" 2>/dev/null || echo 0) + 1 )); echo $n > "$O/.attempts_$key"
      wait_disk "$disk"
      exec 9>"$O/.start.lock"; flock 9                                                           # 시작은 한 번에 하나
      wait_gpu "$gpu"
      log "[w$w] 시작 $name (시도 $n, GPU 여유 $(gpu_free_gb) GB)"
      ( bash -c "$cmd" > "logs/full799_job_${key%.*}.log" 2>&1 ) & local pid=$!
      echo $pid > "$O/.lock_$key/pid"; sleep 90; flock -u 9; exec 9>&-
      wait $pid; local rc=$?
      if [ $rc -eq 0 ] && [ -s "$out" ]; then touch "$done"; log "[w$w] 끝 $name"
      else log "[w$w] 실패 $name (종료 $rc, 결과 $( [ -s "$out" ] && echo 있음 || echo 없음 ))"
        [ $n -ge 2 ] && { echo "$(date '+%F %T') $name" > "$O/.failed_$key"; log "[w$w] 포기 $name"; }; fi
      rm -rf "$O/.lock_$key"; [ "$grp" != gpu ] && rm -rf "$O/.grp_$grp"
      break                                                                                      # 목록을 처음부터 다시 본다
    done < $JOBS
    if [ $remaining -eq 0 ]; then log "[w$w] 남은 작업 없음"; return 0; fi
    [ $ran -eq 0 ] && sleep 60
  done
}
cleanup_locks
# 교체 전에 시작돼 지금 돌고 있는 세 실행은 끝나면 완료로 인정한다(결과를 끝에 쓰는 스크립트)
for k in res_1440000.json student_baseline_v2_epoch_1_native_fp16_nuscenesqa_val.json llm_headroom_kd_only_v4.json; do touch $O/.adopt_$k; done
log "대기열 시작 — 작업자 $K개, 작업 $(wc -l < $JOBS)개"
for w in $(seq 1 $K); do worker $w & sleep 5; done
wait
if ls $O/.failed_* >/dev/null 2>&1; then log "포기한 작업: $(ls $O/.failed_* | xargs -n1 basename | tr '\n' ' ')"; fi
touch $O/.stage_queue_done; log "대기열 완료"
