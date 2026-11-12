#!/bin/bash
# 개정 41 — vast2가 접속되면 SDXL 디코더 bf16(생성 잠재 100개)을 재고 결과를 가져온다. 2026-11-11. 멈추기: touch logs/sdxl_bf16_when_up.stop
cd /home/cj/workspace/vlm_project_2
L=logs/sdxl_bf16_when_up.log; R=/workspace/vlm_project_2; OUT=eval_results/cmp/env3/sdxl_gen_bf16.json
S="ssh -o ConnectTimeout=20 -o BatchMode=yes vast2"
while [ ! -f logs/sdxl_bf16_when_up.stop ]; do
  if $S true 2>/dev/null; then
    echo "$(date '+%F %T') 접속됨" >> $L
    scp -q scripts/sdxl_gen_bf16.py vast2:$R/scripts/sdxl_gen_bf16.py 2>>$L
    $S "cd $R; [ -f $OUT ] || { [ -f logs/sdxl_bf16.pid ] && kill -0 \$(cat logs/sdxl_bf16.pid) 2>/dev/null; } || { HF_HOME=/workspace/.hf_home OMP_NUM_THREADS=4 nohup /workspace/venv_sd/bin/python scripts/sdxl_gen_bf16.py --latents eval_results/cmp/env3/sdxl_gen_cmp.latents.pt --out $OUT > logs/sdxl_bf16.log 2>&1 & echo \$! > logs/sdxl_bf16.pid; }" 2>>$L
    for i in $(seq 1 120); do
      if $S "test -f $R/$OUT" 2>/dev/null; then
        mkdir -p a100_backup/20261111_vast2
        scp -q vast2:$R/$OUT $OUT && scp -q vast2:$R/$OUT vast2:$R/logs/sdxl_bf16.log a100_backup/20261111_vast2/ 2>>$L
        echo "$(date '+%F %T') 끝: $(grep -h '^\[' a100_backup/20261111_vast2/sdxl_bf16.log | tr '\n' ' ')" >> $L; exit 0
      fi
      if ! $S "kill -0 \$(cat $R/logs/sdxl_bf16.pid) 2>/dev/null" 2>/dev/null; then
        $S "test -f $R/$OUT" 2>/dev/null && continue
        echo "$(date '+%F %T') **실패 또는 접속 끊김** — $($S "tail -3 $R/logs/sdxl_bf16.log" 2>/dev/null | tr '\n' ' ')" >> $L; break
      fi
      sleep 60
    done
  else
    echo "$(date '+%F %T') 접속 안 됨" >> $L
  fi
  sleep 180
done
