#!/bin/bash
# 실험 2 실패 계열의 「고친 뒤 다시 잰 값」 — 실험 2 집계(코드 수정 없이 n/4)에는 넣지 않는다 (사전 등록 개정 9).
# 실험 2가 끝나기 전에는 원격 scripts/를 바꾸지 않으려고, 고친 도구를 scripts_retry/ 사본에서 돌린다.
#   nohup bash scripts/fixfam2_retry.sh > logs/fixfam2_retry.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam2
HG="scripts_retry/headroom_guard.py fix --verify --fix_mode auto --limit 799 --fp16_error"
step "재시도 LLaVA-OneVision-0.5B" $O/exp2retry_llava_onevision.json 40 15 \
  $PY $HG --repo llava-hf/llava-onevision-qwen2-0.5b-ov-hf --virtual_ratio 1.5 --out $O/exp2retry_llava_onevision.json
rm -rf "$HF_HOME/hub/models--llava-hf--llava-onevision-qwen2-0.5b-ov-hf"
stage_done fixfam2_retry
