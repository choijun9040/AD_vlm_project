#!/bin/bash
# 서버 쪽 감시 — 끝나지 않았는데 돌고 있지 않은 사슬을 다시 건다(이어 하기 판). 로컬 감시기(watchdog_local.sh)가 3분마다 부른다. 2026-11-05
cd /workspace/vlm_project_2; mkdir -p logs/steps
declare -A DONE=([qwen]=logs/rev30a_qwen.done [gemma]=logs/rev30a_gemma.done [trt]=logs/rev30b.done [idle]=logs/dlm_idle.done [sdxl]=logs/rev31_sdxl.done [sdxl_cococ]=logs/rev33_cococ.done [sdxl_lpips]=logs/rev34_lpips.done [rev35]=logs/rev35.done)
declare -A LEGACY=([qwen]=scripts/rev30a_qwen.sh [gemma]=scripts/rev30a_gemma.sh [trt]=scripts/rev30b_chain.sh [idle]=scripts/dlm_idle_chain.sh [sdxl]=none [sdxl_cococ]=none [sdxl_lpips]=none [rev35]=none)
# 완료 = 표시가 있고 결과 파일이 모두 있을 때. 옛 사슬은 단계가 실패해도 끝에 표시를 남겼다(2026-11-05) — 결과가 빠졌으면 표시를 지우고 새 판으로 다시 건다
O=eval_results/cmp
declare -A NEED=([qwen]="$O/dlm_diag_qwen.json $O/dlm_gen_qwen.json.done $O/dlm_fid_qwen.json $O/llm_headroom_official_dlm799.json"
                 [gemma]="$(for m in 26B-A4B 31B; do for t in 280 1120; do echo -n "$O/dlm_diag_gemma_${m}_$t.json $O/dlm_fid_gemma_${m}_$t.json "; done; done)"
                 [trt]="$O/dlm_trt_a100.json" [idle]="$O/dlm_lat_f16_resid.json $O/dlm_rebuild.json $O/dlm_wrapper_verify.json" [sdxl]="$O/sdxl_coco.json $O/sdxl_enc_resid_plan.json $O/sdxl_coco_full.json" [sdxl_cococ]="$O/sdxl_calib_coco.json $O/sdxl_enc_resid_plan_coco.json $O/sdxl_coco_cococ.json" [sdxl_lpips]="$O/sdxl_lpips_probe.json $O/sdxl_coco_lp01.json" [rev35]="$O/dlm_equiv_qwen.json $O/flux_rev35.json $O/auto100_qwen.json $O/auto100_gemma.json $O/auto100_vae.json $O/auto100_flux.json $O/sdxl_lpips_probe5000.json $O/sdxl_coco_equiv.json $O/dlm_fid_qwen_perimg.json $(for m in 26B-A4B 31B; do for t in 280 1120; do echo -n "$O/dlm_equiv_gemma_${m}_$t.json "; done; done)")
complete(){ [ -f ${DONE[$1]} ] || return 1; for f in ${NEED[$1]}; do [ -f $f ] || { rm -f ${DONE[$1]}; return 1; }; done; }
running(){ ps -eo args | awk -v a="scripts/chains/$1.sh" -v b="${LEGACY[$1]}" '($1=="bash" && ($2==a || $2==b)) {f=1} END {exit !f}'; }
line=""
for c in qwen gemma trt idle sdxl sdxl_cococ sdxl_lpips rev35; do
  if complete $c; then line="$line $c:끝"; continue; fi
  if running $c; then line="$line $c:실행"; continue; fi
  cnt=logs/steps/restart_${c}_$(date +%Y%m%d).count; n=$(cat $cnt 2>/dev/null || echo 0)
  if [ "$n" -ge 6 ]; then line="$line $c:**멈춤(오늘 재시작 $n회 — 사람이 볼 것)**"; continue; fi
  echo $((n+1)) > $cnt
  setsid nohup bash scripts/chains/$c.sh >> logs/chain_$c.log 2>&1 < /dev/null &
  line="$line $c:**재시작($((n+1)))**"
done
gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "$(date '+%F %T') gpu=${gpu}MiB load=$(cut -d' ' -f1 /proc/loadavg)$line"
