#!/bin/bash
# 평가 데이터 전부로 통일 — 작업 대기열 6판 (사전 등록 4fddeb4)
# 6판(2026-10-22): 시작 판단이 앞 작업에 막히지 않게 했다. 5판은 시작 판단 잠금을 쥔 채 예약 자리가 날 때까지 기다렸으므로,
# 40 GB가 필요한 합성 연속체 하나가 기다리는 동안 15 GB면 들어갈 정확도 조각도 그 뒤에서 멈췄다(2026-10-22 12:50 KST — 진행 8개 중
# 실제로 돈 것 2개, GPU 21 GB). 6판은 자리가 없으면 잠금을 풀고 **다음 작업으로 넘어간다**. 큰 작업이 계속 밀리지 않도록,
# 처음 밀린 뒤 STARVE초(기본 1800)가 지난 작업은 잠금을 쥐고 자리가 날 때까지 기다린다(우선권).
# 6판 수정(2026-10-22 15:35 KST): STARVE 기본값 1800 → 21600(6시간). 40 GB 합성 연속체는 한 번에 하나만 들어가므로 다음 합성이 앞 합성이
# 도는 30분 동안 늘 밀려 우선권을 받았고, 우선권 대기가 시작 판단 잠금을 쥐는 동안 15 GB 정확도 조각 4개가 들어갈 자리가 있는데도 서지 못했다
# (15:24 KST부터 GPU에 합성 하나만 돎). 남은 작업은 메모리 총량이 실제 한계이므로 「들어가는 작업부터」가 더 많이 채운다.
# 5판(2026-10-21): vast.ai 점검 뒤 GPU가 「Error 802: system not yet initialized」로 죽은 채 작업이 즉시 실패해 조각 6개가 포기 처리됐다.
# 작업 시작 전에 GPU를 점검하고(죽어 있으면 5분마다 다시 보며 기다린다), GPU·드라이버 오류로 죽은 작업은 시도 횟수에 넣지 않는다.
# 4판(2026-10-21): 묶음 acc(정확도 조각)는 동시에 ACC_MAX개(기본 4)까지 — 정확도가 앞에 있어 예산을 다 잡으면 비전 측정이 하루 넘게 시작하지 못한다.
# 3판(2026-10-21): 예약값을 scripts/full799_resv.txt에서 매번 읽고, 정확도 작업을 목록 앞에 두며, 작업자 기본 8개.
# 비전 측정이 언어 모델을 GPU에 올리지 않게 되어(vision_only_cuda) 정확도 4개를 함께 돌릴 자리가 생겼다.
# --- 이하 2판 설명 ---
# GPU 메모리 **예약** 방식
# 1판은 시작 순간의 여유 메모리만 보았는데, 측정 스크립트는 시작 뒤 수 분에 걸쳐 799장 캐시를 GPU에 쌓으므로
# 802,816 작업 하나가 메모리 부족으로 죽었다(2026-10-21). 2판은 작업마다 **최대 사용량을 예약**하고,
# 돌고 있는 작업의 예약 합 + 새 작업 ≤ 예산일 때만 시작한다. 메모리 부족으로 죽으면 시도 횟수에 넣지 않고 다시 한다.
set -u; source /workspace/vlm_project_2/scripts/full799_lib.sh
K=${1:-8}; BUDGET=${BUDGET:-76}; ACC_MAX=${ACC_MAX:-4}; STARVE=${STARVE:-21600}
JOBS=$O/.jobs.txt; { bash scripts/full799_jobs.sh | grep "_nuscenesqa_val.json|"; bash scripts/full799_jobs.sh | grep -v "_nuscenesqa_val.json|"; } > $JOBS.new && mv $JOBS.new $JOBS   # 정확도(가장 길다)를 앞에
# 예약값(GB) — 2026-10-21 실측(프로세스별 9.6~21.7 GB, 교정 풀기 단독 36.8 GB)에 여유를 둔 값
resv_of(){ local k f p g; k=$(basename "$1")
  while read -r p g; do case "$p" in ""|\#*) continue;; esac; case "$k" in $p) echo $g; return;; esac; done < scripts/full799_resv.txt; echo 24; }
running_pat(){ case "$1" in
  *_nuscenesqa_val.json) local b; b=$(basename "$1"); b=${b%_nuscenesqa_val.json}; local m=${b%%_epoch_1_*}; local t=${b#*_epoch_1_}
    # 조각 작업의 명령은 `--tag native_fp16 --shard 3 --num_shards 8` 꼴이다 — 결과 파일 이름(native_fp16_shard3of8)을 그대로 --tag로 찾으면
    # 맞지 않아 이어받지 못하고 같은 조각을 새로 띄운다(2026-10-22 6판 교체 때 조각 3/8이 두 번 돎, 옛 프로세스를 내려 정리)
    if [[ $t =~ ^(.*)_shard([0-9]+)of([0-9]+)$ ]]; then echo "checkpoints/$m/epoch_1 .*--tag ${BASH_REMATCH[1]} --shard ${BASH_REMATCH[2]} --num_shards ${BASH_REMATCH[3]}\$"
    else echo "checkpoints/$m/epoch_1 .*--tag $t\$"; fi;;
  *) local k=$(basename "$1"); echo "${k%.*}";; esac; }
alive(){ [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
reserved_gb(){ local t=0 d p; for d in $O/.lock_* $O/.resv_*; do [ -d "$d" ] || continue
  p=$(cat "$d/pid" 2>/dev/null); alive "$p" && t=$((t + $(cat "$d/gb" 2>/dev/null || echo 0))); done; echo $t; }
fits(){ [ $(( $(reserved_gb) + $1 )) -le $BUDGET ] && [ "$(gpu_free_gb)" -ge "$1" ]; }
oom_in(){ grep -qE "OutOfMemoryError|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED" "$1" 2>/dev/null; }
infra_in(){ grep -qE "cudaGetDeviceCount|Error 802|system not yet initialized|CUDA driver initialization failed|no CUDA-capable device|CUDA error: unknown error|uncorrectable ECC|CUDA-capable device\(s\) is/are busy" "$1" 2>/dev/null; }
gpu_ok(){ timeout 120 $PY -c "import torch; torch.ones(8, device='cuda').sum().item()" >/dev/null 2>&1; }
wait_gpu_ok(){ local said=0; until gpu_ok; do [ $said = 0 ] && { log "GPU를 쓸 수 없다 — 5분마다 다시 확인하며 기다린다"; said=1; }; sleep 300; done; [ $said = 1 ] && log "GPU 복구 확인"; return 0; }
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
      local gdir=""
      if [ "$grp" = acc ]; then                                              # 정확도: 동시에 ACC_MAX개까지
        local s; for s in $(seq 1 $ACC_MAX); do mkdir "$O/.grp_acc_$s" 2>/dev/null && { gdir="$O/.grp_acc_$s"; break; }; done
        [ -z "$gdir" ] && { rm -rf "$O/.lock_$key"; continue; }; echo $$ > "$gdir/pid"
      elif [ "$grp" != gpu ]; then
        if ! mkdir "$O/.grp_$grp" 2>/dev/null; then rm -rf "$O/.lock_$key"; continue; fi
        gdir="$O/.grp_$grp"; echo $$ > "$gdir/pid"; fi
      ran=1
      local need; need=$(resv_of "$out")
      local n; n=$(( $(cat "$O/.attempts_$key" 2>/dev/null || echo 0) + 1 )); echo $n > "$O/.attempts_$key"
      wait_disk "$disk"
      exec 9>"$O/.start.lock"; flock 9                                        # 시작 판단은 한 번에 하나
      wait_gpu_ok                                                            # GPU가 살아 있을 때만 시작한다
      local ws="$O/.waitsince_$key"
      if ! fits "$need"; then                                                # 예약 합과 실제 여유를 둘 다 본다
        [ -f "$ws" ] || { date +%s > "$ws"; log "[w$w] 자리 없음 — 건너뜀 $name (예약 ${need} GB, 예약 합 $(reserved_gb)/$BUDGET GB)"; }
        if [ $(( $(date +%s) - $(cat "$ws") )) -lt $STARVE ]; then            # 아직 오래 밀리지 않았다 — 잠금을 풀고 다음 작업으로
          flock -u 9; exec 9>&-
          rm -rf "$O/.lock_$key"; [ -n "$gdir" ] && rm -rf "$gdir"
          echo $((n - 1)) > "$O/.attempts_$key"; ran=0
          continue
        fi
        log "[w$w] 우선권 — 자리가 날 때까지 기다린다 $name (${STARVE}초 넘게 밀림)"
        until fits "$need"; do sleep 30; done
      fi
      rm -f "$ws"
      echo $need > "$O/.lock_$key/gb"
      log "[w$w] 시작 $name (시도 $n, 예약 ${need} GB, 예약 합 $(reserved_gb)/$BUDGET GB, 실제 여유 $(gpu_free_gb) GB)"
      local L="logs/full799_job_${key%.*}.log"
      ( bash -c "$cmd" > "$L" 2>&1 ) & local pid=$!
      echo $pid > "$O/.lock_$key/pid"; flock -u 9; exec 9>&-
      wait $pid; local rc=$?
      if [ $rc -eq 0 ] && [ -s "$out" ]; then touch "$done"; log "[w$w] 끝 $name"
      elif infra_in "$L"; then
        echo $((n - 1)) > "$O/.attempts_$key"; cp "$L" "$L.infra$(date +%s)"
        log "[w$w] GPU·드라이버 오류 $name — 시도 횟수에 넣지 않고 GPU 복구를 기다린 뒤 다시 한다"; rm -rf "$O/.lock_$key"; [ -n "$gdir" ] && rm -rf "$gdir"; sleep 60; break
      elif oom_in "$L"; then
        echo $((n - 1)) > "$O/.attempts_$key"; cp "$L" "$L.oom$(date +%s)"
        log "[w$w] 메모리 부족 $name — 시도 횟수에 넣지 않고 다시 한다(5분 뒤)"; rm -rf "$O/.lock_$key"; [ -n "$gdir" ] && rm -rf "$gdir"; sleep 300; break
      else log "[w$w] 실패 $name (종료 $rc, 결과 $( [ -s "$out" ] && echo 있음 || echo 없음 ))"
        [ $n -ge 2 ] && { echo "$(date '+%F %T') $name" > "$O/.failed_$key"; log "[w$w] 포기 $name"; }; fi
      rm -rf "$O/.lock_$key"; [ -n "$gdir" ] && rm -rf "$gdir"
      break
    done < $JOBS
    if [ $remaining -eq 0 ]; then log "[w$w] 남은 작업 없음"; return 0; fi
    [ $ran -eq 0 ] && sleep 60
  done
}
# --- 시작 정리: 미완료 작업의 시도 횟수를 0으로 — 대기열이 다시 뜬 것은 대개 컨테이너 재시작이고, 그때 죽은 실행은 실패가 아니다 ---
for a in $O/.attempts_*; do [ -f "$a" ] || continue; k=${a#$O/.attempts_}; [ -f "$O/.done_$k" ] || rm -f "$a"; done
rm -f $O/.waitsince_*                                                         # 6판: 밀린 시각은 다시 센다
# --- 시작 정리: 살아 있는 실행은 예약을 이어받고, 죽은 잠금은 지운다 ---
for d in $O/.lock_* $O/.grp_*; do [ -d "$d" ] || continue; rm -rf "$d"; done   # 1판 잠금(작업자 pid)은 모두 지우고 아래에서 다시 잡는다
while IFS='|' read -r name out gpu disk grp cmd; do
  key=$(basename "$out"); [ -f "$O/.done_$key" ] && continue
  pat=$(running_pat "$out"); p=$(pgrep -f -- "$pat" | while read q; do grep -q python /proc/$q/cmdline 2>/dev/null && echo $q; done | head -1)
  if [ -n "$p" ]; then mkdir -p "$O/.resv_$key"; echo $p > "$O/.resv_$key/pid"
    g=$(resv_of "$out"); [ $g -lt 14 ] && g=14; echo $g > "$O/.resv_$key/gb"   # 옛 코드로 시작한 실행은 언어 모델까지 GPU에 있다 — 최소 14 GB
    log "이어받음 $name (pid $p, 예약 $(cat $O/.resv_$key/gb) GB)"; fi
done < $JOBS
log "대기열 6판 시작 — 작업자 $K개, 작업 $(wc -l < $JOBS)개, 예산 $BUDGET GB, 지금 예약 합 $(reserved_gb) GB"
for w in $(seq 1 $K); do worker $w & sleep 5; done
wait
if ls $O/.failed_* >/dev/null 2>&1; then log "포기한 작업: $(ls $O/.failed_* | xargs -n1 basename | tr '\n' ' ')"; fi
touch $O/.stage_queue_done; log "대기열 완료"
