#!/bin/bash
# cron에서 5분마다 (/etc/cron.d/fixfam2_resume) — 실험을 띄운 뒤(.armed) 원격 감시가 없으면 다시 띄운다 (2026-11-03).
# 인스턴스가 멈췄다 켜져도 로컬 세션 없이 이어 간다: 감시 → 완료 표시 없는 대기열 → 끝난 단계 건너뜀·중간 저장에서 이어 감
cd /workspace/vlm_project_2 || exit 0
O=eval_results/fixfam2
[ -f $O/.armed ] || exit 0
[ -f $O/.stage_fixfam2_tool_done ] && [ -f $O/.stage_fixfam2_evaltrt_done ] && [ -f $O/.stage_fixfam2_exp5_done ] && exit 0
pgrep -f '[b]ash scripts/fixfam2_watchdog' >/dev/null && exit 0
echo "[$(TZ=Asia/Seoul date '+%F %T KST')] 원격 감시가 없어 다시 띄운다(cron)"
nohup bash scripts/fixfam2_watchdog.sh >> logs/fixfam2_watchdog.log 2>&1 &
