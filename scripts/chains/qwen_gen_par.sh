#!/bin/bash
# 개정 40 ② 단축 (2026-11-10 14:20 KST, 사용자 「단축 검토」): 문장 생성 네 형식을 동시에 — 형식마다 별도 프로세스·별도 out 폴더(par_<형식>).
# 다 끝나면 부분 파일을 원래 이름으로 옮기고 등록한 명령(--runs fp32,bf16,fp16,fp16_resid, c는 측정용 100으로 다시 결정)을 그대로 돌린다 —
# 이어 하기로 생성은 건너뛰고 c 결정·집계만 한다. 병렬 생성에 쓴 c와 집계 단계의 c가 다르면 실패로 멈춘다.
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PY=/workspace/venv_main/bin/python; E=eval_results/cmp/env4
V=data/drivelm_official/val799.jsonl; C=data/drivelm_official/train100.jsonl
CC=0.25    # 측정용 100의 c — 개정 40 ① 같은 측정(81,920 → 1/4), 개정 30 생성 측정과 같음
pids=""
for r in fp32 bf16 fp16 fp16_resid; do
  [ -f logs/steps/g40_$r.done ] && continue
  mkdir -p $E/par_$r; log "  [시작] g40_$r"
  ( $PY scripts/eval_realworldqa.py --q $V --calib_q $C --runs $r --c $CC --max_new_tokens 128 --no_tf32 --out $E/par_$r/dlm_gen_qwen.json >> logs/env4gen_$r.log 2>&1 \
    && touch logs/steps/g40_$r.done && log "  [끝] g40_$r" ) &
  pids="$pids $!"
done
for p in $pids; do wait $p; done
for r in fp32 bf16 fp16 fp16_resid; do [ -f logs/steps/g40_$r.done ] || { log "  **[실패] g40_$r**"; exit 1; }; done
step g40_merge $E/dlm_gen_qwen.json.done -- bash -c "for r in fp32 bf16 fp16 fp16_resid; do cp -n $E/par_\$r/dlm_gen_qwen.json.\$r.partial.jsonl $E/; done
  $PY scripts/eval_realworldqa.py --q $V --calib_q $C --runs fp32,bf16,fp16,fp16_resid --max_new_tokens 128 --no_tf32 --out $E/dlm_gen_qwen.json >> logs/env4gen.log 2>&1 \
  && python3 -c \"import json,sys; c=json.load(open('$E/dlm_gen_qwen.json'))['calib']['c']; print('집계 c', c); sys.exit(0 if c == $CC else 1)\" >> logs/env4gen.log 2>&1 \
  && touch $E/dlm_gen_qwen.json.done" || exit 1
touch logs/env4gen.done; log "사슬 끝"
