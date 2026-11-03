#!/bin/bash
# 교정 확장 2차 실험(개정 7) 감시 — 원격에서 5분마다 (2026-11-01 사용자 요청: 「실험이 중단되는 일 없도록」)
#   nohup bash scripts/fixfam_watchdog.sh > logs/fixfam_watchdog.log 2>&1 &
# 하는 일 — (1) 대기열이 완료 표시 없이 죽으면 다시 띄운다(끝난 단계는 완료 표시로 건너뛰므로 안전, 최대 3회)
#           (2) 디스크 여유가 12GB 아래면 대기열 대상 모델 중 지금 돌지 않는 것의 HF 캐시를 지운다(단계가 끝나면 어차피 지우는 캐시)
#           (3) GPU 사용률 0%가 30분 이어지면 경보만 남긴다(죽이지 않는다)
# 단계 자체의 실패(코드 결함)는 고치지 않는다 — 대기열 로그의 「실패·포기」로 드러난다.
cd /workspace/vlm_project_2; export HF_HOME=/workspace/hf_home
O=eval_results/fixfam2
REPOS="llava-hf/llava-onevision-qwen2-0.5b-ov-hf Qwen/Qwen3-VL-2B-Instruct openbmb/MiniCPM-V-2_6 Salesforce/blip2-opt-2.7b Qwen/Qwen2-VL-2B-Instruct-AWQ HuggingFaceTB/SmolVLM-Instruct ELVISIO/SenseNova-SI-InternVL3-8B-AWQ ybelkada/llava-1.5-7b-hf-awq
Isotr0py/Phi-3.5-vision-instruct-AWQ mistral-community/pixtral-12b zai-org/GLM-4.1V-9B-Thinking moonshotai/Kimi-VL-A3B-Instruct"
w(){ echo "[$(TZ=Asia/Seoul date '+%F %T KST')] [감시] $*"; }
rq=0; re=0; idle=0
w "시작 — 5분 간격"
while true; do
  # (1) 재시작
  if ! pgrep -f '[f]ixfam2_queue.sh' >/dev/null && [ ! -f $O/.stage_fixfam2_tool_done ]; then
    if [ $rq -lt 3 ]; then rq=$((rq+1)); w "대기열이 완료 표시 없이 멈췄다 — 다시 띄운다 ($rq/3)"
      nohup bash scripts/fixfam2_queue.sh >> logs/fixfam2_queue.log 2>&1 &
    else w "경보: 대기열 재시작 3회 초과 — 사람이 볼 것"; fi
  fi
  if ! pgrep -f '[f]ixfam2_eval_trt.sh' >/dev/null && [ ! -f $O/.stage_fixfam2_evaltrt_done ]; then
    if [ $re -lt 3 ]; then re=$((re+1)); w "⑤ 병렬 실행이 완료 표시 없이 멈췄다 — 다시 띄운다 ($re/3)"
      nohup bash scripts/fixfam2_eval_trt.sh >> logs/fixfam2_eval_trt.log 2>&1 &
    else w "경보: ⑤ 재시작 3회 초과 — 사람이 볼 것"; fi
  fi
  if ! pgrep -f '[f]ixfam2_exp5.sh' >/dev/null && [ ! -f $O/.stage_fixfam2_exp5_done ]; then   # 실험 5 (개정 8, 2026-11-03)
    if [ ${r5:-0} -lt 3 ]; then r5=$((${r5:-0}+1)); w "실험 5 대기열을 띄운다 (${r5}/3)"
      nohup bash scripts/fixfam2_exp5.sh >> logs/fixfam2_exp5.log 2>&1 &
    else w "경보: 실험 5 재시작 3회 초과 — 사람이 볼 것"; fi
  fi
  if ! pgrep -f '[f]ixfam2_rtrt.sh' >/dev/null && [ ! -f $O/.stage_fixfam2_rtrt_done ]; then   # (R) TensorRT·Orin 꾸러미 (개정 10)
    if [ ${rr:-0} -lt 3 ]; then rr=$((${rr:-0}+1)); w "(R) TensorRT 대기열을 띄운다 (${rr}/3)"
      nohup bash scripts/fixfam2_rtrt.sh >> logs/fixfam2_rtrt.log 2>&1 &
    else w "경보: (R) TensorRT 재시작 3회 초과 — 사람이 볼 것"; fi
  fi
  if ! pgrep -f '[f]ixfam2_exp5b.sh' >/dev/null && [ ! -f $O/.stage_fixfam2_exp5b_done ]; then   # 실험 5 후속 (P⅛·R⅛)
    if [ ${r5b:-0} -lt 3 ]; then r5b=$((${r5b:-0}+1)); w "실험 5 후속 대기열을 띄운다 (${r5b}/3)"
      nohup bash scripts/fixfam2_exp5b.sh >> logs/fixfam2_exp5b.log 2>&1 &
    else w "경보: 실험 5 후속 재시작 3회 초과 — 사람이 볼 것"; fi
  fi
  # (2) 디스크
  free=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
  if [ "${free:-0}" -lt 12 ]; then
    running=$(ps -eo args | grep -E '[h]eadroom_guard|[r]un_fix_gemma' | grep -o -- '--repo [^ ]*' | awk '{print $2}')
    for r in $REPOS; do
      d="$HF_HOME/hub/models--${r//\//--}"
      [ -d "$d" ] || continue
      grep -qx -- "$r" <<<"$running" && continue
      rm -rf "$d"; w "디스크 여유 ${free}GB — 돌지 않는 모델 캐시 삭제 $r"
    done
    w "디스크 정리 뒤 여유 $(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)GB"
  fi
  # (3) 정지 경보
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -dc 0-9)
  if [ "${u:-0}" -eq 0 ] && { pgrep -f '[f]ixfam2_queue.sh' >/dev/null || pgrep -f '[f]ixfam2_eval_trt.sh' >/dev/null || pgrep -f '[f]ixfam2_exp5.sh' >/dev/null || pgrep -f '[f]ixfam2_rtrt.sh' >/dev/null || pgrep -f '[f]ixfam2_exp5b.sh' >/dev/null; }; then
    idle=$((idle+1)); [ $idle -eq 6 ] && w "경보: 대기열이 살아 있는데 GPU 사용률 0%가 30분 — 대기·정지 확인 필요 (디스크 ${free}GB)"
  else idle=0; fi
  if [ -f $O/.stage_fixfam2_tool_done ] && [ -f $O/.stage_fixfam2_evaltrt_done ] && [ -f $O/.stage_fixfam2_exp5_done ] && [ -f $O/.stage_fixfam2_rtrt_done ] && [ -f $O/.stage_fixfam2_exp5b_done ]; then w "전부 완료 — 감시 종료"; exit 0; fi
  sleep 300
done
