#!/bin/bash
# 교정의 전 계열 확장 — 원격 A100 실행 대기열 (사전 등록: thesis_outline §6 「교정의 전 계열 확장」, 2026-10-30 23:20 KST)
# 공용 함수는 평가 데이터 전부로 통일 때의 것(이어 하기·재시도·GPU/디스크 대기)을 그대로 쓴다. 새 패키지는 설치하지 않는다.
#   nohup bash scripts/fixfam_queue.sh > logs/fixfam_queue.log 2>&1 &
source scripts/full799_lib.sh
O=eval_results/fixfam; mkdir -p $O checkpoints_fix
HG="scripts/headroom_guard.py fix --verify --fix_mode auto --limit 799"
NAT="--max_pixels 1440000 --min_pixels 3136"

# ---- 실제로 상한을 넘는 모델: Qwen2.5-VL (원본 해상도) ----
step "Qwen2.5-VL 기준 모델" $O/qwen25_baseline.json 40 2 \
  $PY $HG --checkpoint checkpoints/student_baseline_v2/epoch_1 $NAT \
  --save_weights checkpoints_fix/baseline_v2_residual_native.pt --out $O/qwen25_baseline.json
step "Qwen2.5-VL 사전학습" $O/qwen25_pretrained.json 40 1 \
  $PY $HG --checkpoint none $NAT --out $O/qwen25_pretrained.json

# ---- 실제로 상한을 넘는 모델: Gemma 4 (블록 뒤) ----
for M in gemma-4-26B-A4B-it gemma-4-31B-it; do for ST in 280 1120; do
  step "Gemma 4 $M $ST" $O/gemma4_${M}_st${ST}.json 40 1 \
    $PG scripts/run_fix_gemma.py --model_dir /workspace/gemma4_vision/$M --max_soft_tokens $ST --limit 799 \
    --out $O/gemma4_${M}_st${ST}.json
done; done

# ---- 상한을 넘지 않는 계열 8: 가상 상한(최악 최대 ÷ 1.5)에서 교정 효과와 등가성 ----
# 디스크 여유가 38GB뿐이라(2026-10-31) 모델마다 측정이 끝나면 HF 캐시를 지운다 — 결과 파일은 남는다
VR="--virtual_ratio 1.5"
rmc(){ rm -rf "$HF_HOME/hub/models--${1//\//--}"; log "캐시 삭제 $1 (디스크 여유 $(disk_free_gb)GB)"; }
step "Qwen2-VL-2B" $O/safe_qwen2vl2b.json 30 6 $PY $HG --repo Qwen/Qwen2-VL-2B-Instruct-AWQ $NAT $VR --out $O/safe_qwen2vl2b.json
rmc Qwen/Qwen2-VL-2B-Instruct-AWQ
step "SmolVLM" $O/safe_smolvlm.json 30 6 $PY $HG --repo HuggingFaceTB/SmolVLM-Instruct $VR --out $O/safe_smolvlm.json
rmc HuggingFaceTB/SmolVLM-Instruct
step "InternVL3-8B" $O/safe_internvl3.json 30 10 $PY $HG --repo ELVISIO/SenseNova-SI-InternVL3-8B-AWQ --trust_remote_code $VR --out $O/safe_internvl3.json
rmc ELVISIO/SenseNova-SI-InternVL3-8B-AWQ
step "LLaVA-1.5-7B" $O/safe_llava.json 30 10 $PY $HG --repo ybelkada/llava-1.5-7b-hf-awq $VR --out $O/safe_llava.json
rmc ybelkada/llava-1.5-7b-hf-awq
step "Phi-3.5-vision" $O/safe_phi35.json 30 6 $PY $HG --repo Isotr0py/Phi-3.5-vision-instruct-AWQ --trust_remote_code $VR --out $O/safe_phi35.json
rmc Isotr0py/Phi-3.5-vision-instruct-AWQ
step "Pixtral-12B" $O/safe_pixtral.json 40 30 $PY $HG --repo mistral-community/pixtral-12b $VR --out $O/safe_pixtral.json
rmc mistral-community/pixtral-12b
step "GLM-4.1V-9B" $O/safe_glm41v.json 40 25 $PG $HG --repo zai-org/GLM-4.1V-9B-Thinking $VR --out $O/safe_glm41v.json
rmc zai-org/GLM-4.1V-9B-Thinking
step "Kimi-VL" $O/safe_kimivl.json 40 40 $PY $HG --repo moonshotai/Kimi-VL-A3B-Instruct --trust_remote_code $VR --out $O/safe_kimivl.json
rmc moonshotai/Kimi-VL-A3B-Instruct
stage_done fixfam_tool

# ---- 기준 ⑤: 기준 모델에 잔차 재조정이 받아들여졌을 때만 — 원본 해상도 fp16 정확도 (11,309문항, 8조각) ----
if $PY -c "import json,sys; d=json.load(open('$O/qwen25_baseline.json')); sys.exit(0 if d.get('residual',{}).get('accepted') and not d.get('applied') else 1)"; then
  for k in 0 1 2 3 4 5 6 7; do
    step "정확도 잔차 재조정 조각 $k/8" $O/student_baseline_v2_epoch_1_native_fp16_resid_shard${k}of8_nuscenesqa_val.json 20 1 \
      $PY scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --dump_predictions --log_every 1000 \
      --out_dir $O $NAT --lora_dtype float16 --vision_state checkpoints_fix/baseline_v2_residual_native.pt \
      --tag native_fp16_resid --shard $k --num_shards 8
  done
  $PY scripts/compare_fix_predictions.py --fix "$O/student_baseline_v2_epoch_1_native_fp16_resid_shard*of8_predictions.jsonl" \
    --ref "eval_results/full799/student_baseline_v2_epoch_1_native_bf16_shard*of8_predictions.jsonl" --out $O/criterion5.json
else
  log "기준 ⑤ 건너뜀 — 기준 모델에 잔차 재조정만으로 교정되지 않았다(사전 등록: H1이 성립할 때만)"
fi
stage_done fixfam_all
