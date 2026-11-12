# 사슬 공용 — 단계별 완료 표시(logs/steps/<이름>.done). 끊겨 다시 걸면 끝난 단계는 건너뛴다. 2026-11-05
mkdir -p logs/steps
log(){ echo "[$(date '+%F %T')] $*"; }
# step 이름 [최종 결과 파일 — 있으면 끝난 것으로 본다(그 스크립트가 다 끝난 뒤에만 쓰는 파일만)] -- 명령...
step(){
  local name=$1; shift; local fin=""
  if [ "$1" != "--" ]; then fin=$1; shift; fi; shift
  if [ -f logs/steps/$name.done ] || { [ -n "$fin" ] && [ -f "$fin" ]; }; then log "  [건너뜀] $name"; touch logs/steps/$name.done; return 0; fi
  log "  [시작] $name"
  if "$@"; then touch logs/steps/$name.done; log "  [끝] $name"; return 0; fi
  log "  **[실패] $name**"; return 1
}
