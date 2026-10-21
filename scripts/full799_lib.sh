# 평가 데이터 전부로 통일 — 단계 스크립트 공용 함수 (사전 등록 4fddeb4)
# 이어 하기: 결과 파일이 있으면 건너뛴다. 중복 방지: 같은 결과를 쓰는 프로세스가 돌고 있으면 끝날 때까지 기다린다.
# 재시도: 실패하면 한 번 더 한다. 자원: GPU 여유 메모리·디스크가 모자라면 기다린다.
cd /workspace/vlm_project_2; export HF_HOME=/workspace/hf_home
PY=/workspace/venv_main/bin/python; PG=/workspace/venv_gemma/bin/python; PT=/workspace/venv_trt/bin/python
O=eval_results/full799; mkdir -p $O logs
log(){ echo "[$(date '+%F %T')] $*"; }
gpu_free_gb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
disk_free_gb(){ df -BG /workspace | tail -1 | awk '{gsub("G","",$4); print $4}'; }
wait_gpu(){ local need=$1; until [ "$(gpu_free_gb)" -ge "$need" ]; do sleep 60; done; }
wait_disk(){ local need=$1; until [ "$(disk_free_gb)" -ge "$need" ]; do sleep 120; done; }
# step <이름> <결과 파일> <GPU GB> <디스크 GB> <명령...>
step(){
  local name=$1 out=$2 gpu=$3 disk=$4; shift 4
  local key; key=$(basename "$out")
  local done="$O/.done_$key"
  local stem="${key%.*}"   # 중복 실행 확인은 확장자 없는 이름으로 — 명령줄에는 이미지별 파일(<이름>_per_image.jsonl)만 들어 있는 경우가 있다
  if [ -f "$done" ]; then log "건너뜀 $name (완료 표시 있음)"; return 0; fi
  # 같은 결과를 쓰는 실행이 이미 돌고 있으면(오케스트레이터 교체 전에 시작된 것) 끝날 때까지 기다리고,
  # 그 실행이 결과를 남겼으면 완료로 본다. **결과를 끝에 한 번만 쓰는 스크립트에만 맞는 규칙**이므로
  # 이 경우는 로그에 따로 남긴다.
  if pgrep -f -- "$stem" >/dev/null; then
    log "대기 $name — 이미 돌고 있는 실행이 끝나기를"
    while pgrep -f -- "$stem" >/dev/null; do sleep 60; done
    if [ -s "$out" ]; then touch "$done"; log "완료로 인정 $name (앞선 실행의 결과)"; return 0; fi
  fi
  for attempt in 1 2; do
    wait_gpu "$gpu"; wait_disk "$disk"
    log "시작 $name (시도 $attempt)"
    "$@" 2>&1 | grep --line-buffered -E "^\[|저장|→|이미지별|Error|Traceback|error|상실|여유|무효|정확도|Overall" | tail -25
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ] && [ -s "$out" ]; then touch "$done"; log "끝 $name"; rm -f "$O/.failed_$key"; return 0; fi
    log "실패 $name (종료 $rc, 결과 $( [ -s "$out" ] && echo 있음 || echo 없음 ))"; sleep 30
  done
  echo "$(date '+%F %T') $name" > "$O/.failed_$key"; log "포기 $name — $O/.failed_$key 에 기록하고 다음으로"
  return 1
}
stage_done(){ touch "$O/.stage_$1_done"; log "$1 단계 완료"; }
