#!/bin/bash
# 새 인스턴스(vast2) 전용 감시 — 개정 35~37 사슬(rev35.sh)과 개정 38 사슬(sdxl_env2.sh)을 본다. 2026-11-09
# 끝나지 않았는데 돌고 있지 않으면 다시 건다(횟수 한도 없음 — 사용자 「한도 제거」, 횟수는 기록).
# 멈춤 감지: 돌고 있는데 그 사슬의 진행 파일이 90분 동안 하나도 바뀌지 않으면 사슬 프로세스 그룹을 끊고 다시 건다.
# 옛 supervise.sh는 쓰지 않는다 — 새 인스턴스에는 옛 사슬의 완료 표시가 없어 모두 다시 시작하기 때문이다.
cd /workspace/vlm_project_2; mkdir -p logs/steps
O=eval_results/cmp; E=$O/env2; STALL_MIN=90
declare -A DONE=([rev35]=logs/rev35.done [sdxl_env2]=logs/env2.done [sdxl_tf32off]=logs/env3.done [qwen_tf32off]=logs/env4.done [qwen_gen_par]=logs/env4gen.done)
declare -A NEED=([rev35]="$O/dlm_equiv_qwen.json $(for m in 26B-A4B 31B; do for t in 280 1120; do echo -n "$O/dlm_equiv_gemma_${m}_$t.json "; done; done)$O/auto100_qwen.json $O/auto100_gemma.json $O/auto100_vae.json $O/auto100_flux.json $O/sdxl_lpips_probe5000.json $O/sdxl_coco_equiv.json $O/dlm_fid_qwen_perimg.json $O/flux_rev35.json"
                 [sdxl_env2]="$E/sdxl_calib.json $E/sdxl_gen_cmp.json $E/sdxl_gen_view.png $E/sdxl_enc_resid_plan_coco.json $E/sdxl_calib_coco.json $E/sdxl_enc_resid_plan.json $E/sdxl_coco.json $E/sdxl_coco_genc.json $E/sdxl_coco_equiv.json"
                 [sdxl_tf32off]="$O/env3/sdxl_gen_cmp.json $O/env3/sdxl_coco.json $O/env3/sdxl_coco_genc.json"
                 [qwen_tf32off]="$O/env4/dlm_fid_qwen.json"
                 [qwen_gen_par]="$O/env4/dlm_gen_qwen.json.done")   # 2026-11-10 14:20: 문장 생성은 qwen_gen_par로 옮김(병렬)
# 사슬마다 진행 파일(로그·결과) — 멈춤 판정용
declare -A PROG=([rev35]="logs/chain_rev35.log logs/rev35_equiv_qwen.log logs/rev35_equiv_gemma.log logs/rev36_auto.log logs/rev37.log logs/rev35_flux.log $O" [sdxl_env2]="logs/chain_sdxl_env2.log logs/env2.log $E /dev/shm/coco" [sdxl_tf32off]="logs/chain_sdxl_tf32off.log logs/env3.log $O/env3 /dev/shm/coco" [qwen_tf32off]="logs/chain_qwen_tf32off.log logs/env4.log $O/env4" [qwen_gen_par]="logs/chain_qwen_gen_par.log logs/env4gen.log $(for r in fp32 bf16 fp16 fp16_resid; do echo -n "logs/env4gen_$r.log $O/env4/par_$r "; done)$O/env4")
done_all(){ [ -f ${DONE[$1]} ] || return 1; for f in ${NEED[$1]}; do [ -e $f ] || { rm -f ${DONE[$1]}; return 1; }; done; }
pid_of(){ ps -eo pid,args | awk -v s="scripts/chains/$1.sh" '$2=="bash" && $3==s {print $1; exit}'; }
# 2026-11-10: 형식별로 한꺼번에 저장하는 단계(Qwen 보존·문장 생성)는 90분 넘게 파일이 그대로일 수 있다 —
# 사슬 프로세스 그룹($2)의 프로세스가 GPU SM을 5% 이상 쓰고 있으면 멈춤이 아니다.
gpu_busy(){ local pids; pids=$(ps -eo pid=,pgid= | awk -v g="$1" '$2==g {print $1}'); [ -n "$pids" ] || return 1
            timeout 15 nvidia-smi pmon -c 2 -s u 2>/dev/null | awk -v P=" $(echo $pids) " '$1!~/^#/ && index(P," "$2" ") && $4+0>=5 {f=1} END {exit !f}'; }
stalled(){ [ -z "$(find ${PROG[$1]} -maxdepth 1 -newermt "-${STALL_MIN} min" 2>/dev/null | head -1)" ] && ! gpu_busy "$2"; }
start(){ cnt=logs/steps/restart_$1_$(date +%Y%m%d).count; n=$(cat $cnt 2>/dev/null || echo 0); echo $((n+1)) > $cnt
         setsid nohup bash scripts/chains/$1.sh >> logs/chain_$1.log 2>&1 < /dev/null & echo "오늘 $((n+1))회째"; }
line=""
for c in rev35 sdxl_env2 sdxl_tf32off qwen_tf32off qwen_gen_par; do
  p=$(pid_of $c)
  if done_all $c; then line="$line $c:끝"
  elif [ -n "$p" ] && ! stalled $c $p; then line="$line $c:실행"
  elif [ -n "$p" ]; then
    echo "$(date '+%F %T') **멈춤 감지 — ${STALL_MIN}분 동안 진행 파일 변화 없음, 사슬 그룹 $p 종료 후 재시작**" >> logs/chain_$c.log
    kill -TERM -- -$p 2>/dev/null; sleep 10; kill -KILL -- -$p 2>/dev/null
    line="$line $c:**멈춤 감지 → 재시작($(start $c))**"
  else line="$line $c:**재시작($(start $c))**"; fi
done
gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "$(date '+%F %T') gpu=${gpu} load=$(cut -d' ' -f1 /proc/loadavg)$line"
