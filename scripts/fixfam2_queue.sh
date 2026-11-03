#!/bin/bash
# 사전 등록 개정 7 — 실험 2(처음 보는 4계열, 코드 수정 없이)와 실험 4(fp16 오차, 교정 전·후). 원격 A100.
#   nohup bash scripts/fixfam2_queue.sh > logs/fixfam2_queue.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam2; mkdir -p $O
HG="scripts/headroom_guard.py fix --verify --fix_mode auto --limit 799 --fp16_error"
NAT="--max_pixels 1440000 --min_pixels 3136"; VR="--virtual_ratio 1.5"
rmc(){ rm -rf "$HF_HOME/hub/models--${1//\//--}"; log "캐시 삭제 $1 (디스크 여유 $(disk_free_gb)GB)"; }
fam(){ local name=$1 out=$2 py=$3 repo=$4; shift 4
  step "$name" $O/$out.json 40 15 $py $HG --repo $repo "$@" --out $O/$out.json; rmc $repo; }

# ── 실험 2: 처음 보는 4계열 (실험 4의 fp16 오차도 같은 실행에서) ──────────────────────
fam "실험2 LLaVA-OneVision-0.5B" exp2_llava_onevision $PY llava-hf/llava-onevision-qwen2-0.5b-ov-hf $VR
fam "실험2 Qwen3-VL-2B" exp2_qwen3vl2b $PG Qwen/Qwen3-VL-2B-Instruct $NAT $VR
fam "실험2 MiniCPM-V-2.6" exp2_minicpmv26 $PY openbmb/MiniCPM-V-2_6 --trust_remote_code $VR
fam "실험2 BLIP-2 OPT-2.7B" exp2_blip2 $PY Salesforce/blip2-opt-2.7b $VR
stage_done fixfam2_exp2

# ── 실험 4: 기존 안전 8계열과 Qwen2.5-VL 기준 모델 — 교정 전·후 fp16 오차 ──────────────────
fam "실험4 SmolVLM" exp4_smolvlm $PY HuggingFaceTB/SmolVLM-Instruct $VR
fam "실험4 InternVL3-8B" exp4_internvl3 $PY ELVISIO/SenseNova-SI-InternVL3-8B-AWQ --trust_remote_code $VR
fam "실험4 LLaVA-1.5-7B" exp4_llava $PY ybelkada/llava-1.5-7b-hf-awq $VR
fam "실험4 Phi-3.5-vision" exp4_phi35 $PY Isotr0py/Phi-3.5-vision-instruct-AWQ --trust_remote_code $VR
fam "실험4 Pixtral-12B" exp4_pixtral $PY mistral-community/pixtral-12b $VR
fam "실험4 Kimi-VL" exp4_kimivl $PY moonshotai/Kimi-VL-A3B-Instruct --trust_remote_code $VR
fam "실험4 GLM-4.1V-9B" exp4_glm41v $PG zai-org/GLM-4.1V-9B-Thinking $NAT $VR
fam "실험4 Qwen2-VL-2B" exp4_qwen2vl2b $PY Qwen/Qwen2-VL-2B-Instruct-AWQ $NAT $VR
step "실험4 Qwen2.5-VL 기준 모델" $O/exp4_qwen25_baseline.json 40 2 \
  $PY $HG --checkpoint checkpoints/student_baseline_v2/epoch_1 $NAT --out $O/exp4_qwen25_baseline.json
stage_done fixfam2_exp4
stage_done fixfam2_tool
