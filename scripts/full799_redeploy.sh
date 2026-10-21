#!/bin/bash
# 서버(vast.ai A100)가 점검·재시작에서 돌아온 뒤 측정 대기열을 5판(GPU 점검 포함)으로 맞춘다 — 로컬에서 실행한다.
# 1분마다 접속을 시도하고, 접속되면 옛 감시를 내리고, GPU 오류로 생긴 포기 표시를 지우고, 감시 full799f를 올린 뒤 GPU를 점검한다.
#   bash scripts/full799_redeploy.sh          (주소·포트가 바뀌면 HOST·PORT 환경 변수로 준다)
R="ssh -o BatchMode=yes -o ConnectTimeout=10 -p ${PORT:?PORT를 주세요} root@${HOST:?HOST를 주세요}"
until timeout 20 $R 'true' >/dev/null 2>&1; do sleep 60; done
echo "접속 복구 $(date '+%F %T KST')"
sleep 20
timeout 300 $R 'cd /workspace/vlm_project_2
supervisorctl stop full799e full799d 2>/dev/null; rm -f /etc/supervisor/conf.d/full799e.conf /etc/supervisor/conf.d/full799d.conf; supervisorctl reread >/dev/null; supervisorctl update >/dev/null
for p in $(ps -eo pid,args | awk "\$2==\"bash\" && \$3 ~ /^scripts\/(full799_q|run_full799)/ {print \$1}"); do kill $p; done
for p in $(ps -eo pid,args | awk "\$2 ~ /venv_(main|gemma)\/bin\/python/ && \$3 ~ /^scripts\// {print \$1}"); do kill $p; done
git pull -q; git log --oneline -1
O=eval_results/full799
for f in $O/.failed_*; do [ -f "$f" ] || continue; k=${f#$O/.failed_}; L=logs/full799_job_${k%.*}.log; if grep -qE "Error 802|cudaGetDeviceCount" $L 2>/dev/null; then rm -f $f $O/.attempts_$k; echo "포기 표시 지움: $k"; else echo "남김: $k"; fi; done
cp scripts/full799_supervisor6.conf /etc/supervisor/conf.d/full799f.conf; supervisorctl reread >/dev/null; supervisorctl update
echo "--- GPU 점검"; timeout 120 /workspace/venv_main/bin/python -c "import torch; print(\"GPU 연산 성공\", torch.ones(8,device=\"cuda\").sum().item())" 2>&1 | tail -1
ps -o lstart= -p 1; sleep 60; supervisorctl status | grep -E "full799|cgroup"; tail -4 logs/full799_queue.log | cut -c1-130' 2>&1 | grep -v "Welcome\|Have fun\|setlocale"
