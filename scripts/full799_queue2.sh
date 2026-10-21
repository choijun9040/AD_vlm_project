#!/bin/bash
# 평가 데이터 전부로 통일 — 작업 대기열 2판: GPU 메모리 **예약** 방식 (사전 등록 4fddeb4)
# 1판은 시작 순간의 여유 메모리만 보았는데, 측정 스크립트는 시작 뒤 수 분에 걸쳐 799장 캐시를 GPU에 쌓으므로
# 802,816 작업 하나가 메모리 부족으로 죽었다(2026-10-21). 2판은 작업마다 **최대 사용량을 예약**하고,
# 돌고 있는 작업의 예약 합 + 새 작업 ≤ 예산일 때만 시작한다. 메모리 부족으로 죽으면 시도 횟수에 넣지 않고 다시 한다.
set -u; source /workspace/vlm_project_2/scripts/full799_lib.sh
K=${1:-5}; BUDGET=${BUDGET:-76}
JOBS=$O/.jobs.txt; bash scripts/full799_jobs.sh > $JOBS.new && mv $JOBS.new $JOBS
# 예약값(GB) — 2026-10-21 실측(프로세스별 9.6~21.7 GB, 교정 풀기 단독 36.8 GB)에 여유를 둔 값
resv_of(){ local o=$1 k; k=$(basename "$1"); case "$k" in
  res_1440000*|res_*1440000*) echo 18;; res_802816__*) echo 13;; res_401408__*) echo 11;; res_200704__*|res_100352__*) echo 10;;
  v1_vs_v2*|align_target*|training_trajectory__*) echo 10;;
  *_native_*nuscenesqa_val.json) echo 18;; *_eval_*nuscenesqa_val.json) echo 14;;
  llm_headroom_*) echo 28;;
  guard_sweep_align*|guard_dla_demo*|headroom_guard_fix_base3b*|detection_curve__*) echo 40;;
  int8_quant_error_native*|mlp_overflow*|fixa_weight_equiv*|channel849*) echo 30;;
  blk31_*|norm2_direction_*) echo 14;;
  gemma4_vision_*) echo 16;; gemma4_e2e_*) echo 62;; altimages_CAM_BACK*) echo 40;;
  guard_qwen2vl_*|headroom_guard_smolvlm*|natural_collapse*|other_vlm_families*|int8_across_families*|vision_tower_direct.json|vision_tower_direct_official*|vision_tower_inheritance*) echo 32;;
  vision_tower_direct_pixtral*) echo 36;; vision_tower_direct_kimi*|bridge_newhf*) echo 48;;
  *) echo 24;; esac; }
running_pat(){ case "$1" in
  *_nuscenesqa_val.json) local b; b=$(basename "$1"); b=${b%_nuscenesqa_val.json}; local m=${b%%_epoch_1_*}; local t=${b#*_epoch_1_}; echo "checkpoints/$m/epoch_1 .*--tag $t\$";;
  *) local k=$(basename "$1"); echo "${k%.*}";; esac; }
alive(){ [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
reserved_gb(){ local t=0 d p; for d in $O/.lock_* $O/.resv_*; do [ -d "$d" ] || continue
  p=$(cat "$d/pid" 2>/dev/null); alive "$p" && t=$((t + $(cat "$d/gb" 2>/dev/null || echo 0))); done; echo $t; }
oom_in(){ grep -qE "OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED" "$1" 2>/dev/null; }
worker(){
  local w=$1
  while true; do
    local remaining=0 ran=0
    while IFS='|' read -r name out gpu disk grp cmd; do
      local key; key=$(basename "$out"); local done=$O/.done_$key
      [ -f "$done" ] && continue
      [ -f "$O/.failed_$key" ] && continue
      remaining=1
      if [ -d "$O/.resv_$key" ]; then                                       # 이어받은 실행(교체 전에 시작된 것)
        alive "$(cat $O/.resv_$key/pid 2>/dev/null)" && continue
        rm -rf "$O/.resv_$key"
        if [ -s "$out" ]; then touch "$done"; log "[w$w] 완료로 인정 $name (이어받은 실행의 결과)"; continue; fi
        log "[w$w] 이어받은 실행이 결과 없이 끝남 — 다시 한다: $name"
      fi
      mkdir "$O/.lock_$key" 2>/dev/null || continue
      echo $$ > "$O/.lock_$key/pid"
      if [ "$grp" != gpu ]; then
        if ! mkdir "$O/.grp_$grp" 2>/dev/null; then rm -rf "$O/.lock_$key"; continue; fi
        echo $$ > "$O/.grp_$grp/pid"; fi
      ran=1
      local need; need=$(resv_of "$out")
      local n; n=$(( $(cat "$O/.attempts_$key" 2>/dev/null || echo 0) + 1 )); echo $n > "$O/.attempts_$key"
      wait_disk "$disk"
      exec 9>"$O/.start.lock"; flock 9                                        # 시작 판단은 한 번에 하나
      until [ $(( $(reserved_gb) + need )) -le $BUDGET ] && [ "$(gpu_free_gb)" -ge 4 ]; do sleep 30; done
      echo $need > "$O/.lock_$key/gb"
      log "[w$w] 시작 $name (시도 $n, 예약 ${need} GB, 예약 합 $(reserved_gb)/$BUDGET GB, 실제 여유 $(gpu_free_gb) GB)"
      local L="logs/full799_job_${key%.*}.log"
      ( bash -c "$cmd" > "$L" 2>&1 ) & local pid=$!
      echo $pid > "$O/.lock_$key/pid"; flock -u 9; exec 9>&-
      wait $pid; local rc=$?
      if [ $rc -eq 0 ] && [ -s "$out" ]; then touch "$done"; log "[w$w] 끝 $name"
      elif oom_in "$L"; then
        echo $((n - 1)) > "$O/.attempts_$key"; cp "$L" "$L.oom$(date +%s)"
        log "[w$w] 메모리 부족 $name — 시도 횟수에 넣지 않고 다시 한다(5분 뒤)"; rm -rf "$O/.lock_$key"; [ "$grp" != gpu ] && rm -rf "$O/.grp_$grp"; sleep 300; break
      else log "[w$w] 실패 $name (종료 $rc, 결과 $( [ -s "$out" ] && echo 있음 || echo 없음 ))"
        [ $n -ge 2 ] && { echo "$(date '+%F %T') $name" > "$O/.failed_$key"; log "[w$w] 포기 $name"; }; fi
      rm -rf "$O/.lock_$key"; [ "$grp" != gpu ] && rm -rf "$O/.grp_$grp"
      break
    done < $JOBS
    if [ $remaining -eq 0 ]; then log "[w$w] 남은 작업 없음"; return 0; fi
    [ $ran -eq 0 ] && sleep 60
  done
}
# --- 시작 정리: 살아 있는 실행은 예약을 이어받고, 죽은 잠금은 지운다 ---
for d in $O/.lock_* $O/.grp_*; do [ -d "$d" ] || continue; rm -rf "$d"; done   # 1판 잠금(작업자 pid)은 모두 지우고 아래에서 다시 잡는다
while IFS='|' read -r name out gpu disk grp cmd; do
  key=$(basename "$out"); [ -f "$O/.done_$key" ] && continue
  pat=$(running_pat "$out"); p=$(pgrep -f -- "$pat" | while read q; do grep -q python /proc/$q/cmdline 2>/dev/null && echo $q; done | head -1)
  if [ -n "$p" ]; then mkdir -p "$O/.resv_$key"; echo $p > "$O/.resv_$key/pid"; resv_of "$out" > "$O/.resv_$key/gb"
    log "이어받음 $name (pid $p, 예약 $(cat $O/.resv_$key/gb) GB)"; fi
done < $JOBS
log "대기열 2판 시작 — 작업자 $K개, 작업 $(wc -l < $JOBS)개, 예산 $BUDGET GB, 지금 예약 합 $(reserved_gb) GB"
for w in $(seq 1 $K); do worker $w & sleep 5; done
wait
if ls $O/.failed_* >/dev/null 2>&1; then log "포기한 작업: $(ls $O/.failed_* | xargs -n1 basename | tr '\n' ' ')"; fi
touch $O/.stage_queue_done; log "대기열 완료"
