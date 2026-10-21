#!/bin/bash
# 평가 데이터 전부로 통일 — 3단계(이어 하기 판): 범위 확장 A·B, 도구 검증, 다른 계열·공개 저장소, 검증 CAM_BACK (사전 등록 4fddeb4)
set -u; source /workspace/vlm_project_2/scripts/full799_lib.sh
for m in pretrained baseline_v2 kd_only_v4 temporal spatial full; do
  ck=checkpoints/student_$m/epoch_1; [ $m = pretrained ] && ck=none
  step "B $m" $O/llm_headroom_$m.json 16 1 $PY scripts/probe_llm_headroom.py --checkpoint $ck --tag $m --limit 799 --out_dir $O --per_image $O/llm_headroom_${m}_per_image.jsonl
done
for m in gemma-4-26B-A4B-it gemma-4-31B-it; do for st in 280 1120; do
  step "A $m st$st" $O/gemma4_vision_${m}_st${st}.json 16 1 $PG scripts/probe_gemma4_vision.py --model_dir /workspace/gemma4_vision/$m --max_soft_tokens $st --limit 799 --out_dir $O \
      --per_image $O/gemma4_vision_${m}_st${st}_per_image.jsonl
done; done
step "A 종단 799" $O/gemma4_e2e_26B-A4B_full799.json 60 1 $PG scripts/probe_gemma4_e2e.py --offset 0 --limit 799 --out $O/gemma4_e2e_26B-A4B_full799.json
[ -f $O/.done_gemma4_e2e_26B-A4B_full799.json ] && rm -rf /workspace/hf_home/hub/models--google--gemma-4-26B-A4B-it && log "Gemma 26B 캐시 삭제 (디스크 확보)"
G=( --dtype float16 --limit 799 )
step "도구 Qwen2-VL-2B-AWQ" $O/guard_qwen2vl_2b_awq.json 20 15 $PY scripts/headroom_guard.py diagnose --repo Qwen/Qwen2-VL-2B-Instruct-AWQ "${G[@]}" --max_pixels 1440000 --out $O/guard_qwen2vl_2b_awq.json --per_image $O/guard_qwen2vl_2b_awq_per_image.jsonl
step "도구 Qwen2-VL-7B-AWQ" $O/guard_qwen2vl_7b_awq.json 24 20 $PY scripts/headroom_guard.py diagnose --repo Qwen/Qwen2-VL-7B-Instruct-AWQ "${G[@]}" --max_pixels 1440000 --out $O/guard_qwen2vl_7b_awq.json --per_image $O/guard_qwen2vl_7b_awq_per_image.jsonl
step "도구 SmolVLM" $O/headroom_guard_smolvlm.json 20 15 $PY scripts/headroom_guard.py diagnose --repo HuggingFaceTB/SmolVLM-Instruct "${G[@]}" --out $O/headroom_guard_smolvlm.json --per_image $O/headroom_guard_smolvlm_per_image.jsonl
step "도구 기준선·DLA" $O/guard_dla_demo.json 24 2 $PY scripts/headroom_guard.py diagnose --checkpoint checkpoints/student_baseline_v2/epoch_1 "${G[@]}" --max_pixels 1440000 --available_formats float16,int8 --out $O/guard_dla_demo.json --per_image $O/guard_dla_demo_per_image.jsonl
step "도구 사전학습 3B fix" $O/headroom_guard_fix_base3b.json 24 2 $PY scripts/headroom_guard.py fix --checkpoint none "${G[@]}" --max_pixels 1440000 --target_headroom 2.0 --verify --out $O/headroom_guard_fix_base3b.json --per_image $O/headroom_guard_fix_base3b_per_image.jsonl
step "자연 상실" $O/natural_collapse.json 24 25 $PY scripts/probe_natural_collapse.py --limit 799 --max_pixels 1440000 --cleanup --out $O/natural_collapse.json --per_image $O/natural_collapse_per_image.jsonl
step "다른 계열 프로파일" $O/other_vlm_families_profile.json 24 25 $PY scripts/profile_other_vlm_families.py --limit 799 --max_pixels 1440000 --cleanup --out $O/other_vlm_families_profile.json --per_image $O/other_vlm_families_per_image.jsonl
step "다른 계열 INT8" $O/int8_across_families.json 24 25 $PY scripts/probe_int8_across_families.py --limit 799 --max_pixels 1440000 --cleanup --out $O/int8_across_families.json --per_image $O/int8_across_families_per_image.jsonl
step "직접 AngelSlim" $O/vision_tower_direct.json 24 25 $PY scripts/probe_vision_tower_direct.py --repos AngelSlim/Qwen2.5-VL-3B-Instruct-INT4-AWQ AngelSlim/Qwen2.5-VL-7B-Instruct-AWQ --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct.json --per_image $O/vision_tower_direct_per_image.jsonl
step "직접 공식" $O/vision_tower_direct_official.json 24 25 $PY scripts/probe_vision_tower_direct.py --repos Qwen/Qwen2.5-VL-3B-Instruct-AWQ Qwen/Qwen2.5-VL-7B-Instruct-AWQ Qwen/Qwen2.5-VL-3B-Instruct --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct_official.json --per_image $O/vision_tower_direct_official_per_image.jsonl
step "직접 상속 쌍" $O/vision_tower_inheritance.json 24 30 $PY scripts/probe_vision_tower_direct.py --repos llava-hf/llava-1.5-7b-hf ybelkada/llava-1.5-7b-hf-awq microsoft/Phi-3.5-vision-instruct Isotr0py/Phi-3.5-vision-instruct-AWQ sensenova/SenseNova-SI-InternVL3-8B ELVISIO/SenseNova-SI-InternVL3-8B-AWQ --trust_remote_code --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_inheritance.json --per_image $O/vision_tower_inheritance_per_image.jsonl
step "직접 Pixtral" $O/vision_tower_direct_pixtral.json 32 30 $PY scripts/probe_vision_tower_direct.py --repos mistral-community/pixtral-12b --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct_pixtral.json --per_image $O/vision_tower_direct_pixtral_per_image.jsonl
step "직접 Kimi" $O/vision_tower_direct_kimi.json 40 40 $PY scripts/probe_vision_tower_direct.py --repos moonshotai/Kimi-VL-A3B-Instruct --trust_remote_code --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct_kimi.json --per_image $O/vision_tower_direct_kimi_per_image.jsonl
[ -s $O/bridge_image_paths.json ] || $PY scripts/make_image_paths.py --root /workspace/vlm_project_2 --n 799 --out $O/bridge_image_paths.json
step "다리 측정(새 transformers)" $O/bridge_newhf.json 40 40 $PG scripts/bridge_measure_newhf.py --repos Qwen/Qwen2.5-VL-3B-Instruct QuantTrio/Qwen3-VL-32B-Instruct-AWQ QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ zai-org/GLM-4.1V-9B-Thinking moonshotai/Kimi-VL-A3B-Instruct \
    --paths $O/bridge_image_paths.json --max_pixels 1440000 --cleanup --out $O/bridge_newhf.json --per_image $O/bridge_newhf_per_image.jsonl
step "CAM_BACK 검증 799" $O/altimages_CAM_BACK_val799.json 30 2 $PY scripts/profile_alt_images.py --camera CAM_BACK --val_only --limit 799 --min_pixels 3136 --max_pixels 1440000 \
    --only base_3b student_baseline_v2 student_kd_only_v4 student_temporal student_spatial student_full teacher_7b \
    --out $O/altimages_CAM_BACK_val799.json --per_image $O/altimages_CAM_BACK_val799_per_image.jsonl
stage_done s3r
