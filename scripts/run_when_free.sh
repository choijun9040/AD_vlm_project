# 사용법: bash scripts/run_when_free.sh <필요 MiB> <완료 표시 파일> <로그> <명령...> — GPU 여유가 생기면 명령을 돌리고, 완료 파일이 생길 때까지 되풀이(메모리 부족 대비, 2026-11-04)
need=$1; done_file=$2; logf=$3; shift 3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
until [ -f "$done_file" ]; do
  free=$((81920 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)))
  if [ $free -ge $need ]; then echo "[$(date "+%F %T")] 여유 $free MiB — 시작: $*"; "$@" >> "$logf" 2>&1 < /dev/null; echo "[$(date "+%F %T")] 종료 $?"; fi
  sleep 120
done
echo "[$(date "+%F %T")] 완료 파일 있음 — 끝"
