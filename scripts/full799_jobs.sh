#!/bin/bash
# 평가 데이터 전부로 통일 — 작업 목록 (사전 등록 4fddeb4). 한 줄 = 이름|결과 파일|GPU GB|디스크 GB|묶음|명령
# GPU GB는 799장 전처리 캐시를 GPU에 올리는 스크립트 기준(2026-10-21 메모리 부족 1건 뒤 상향).
# 묶음: gpu(아무 때나 병렬) · dl(공개 저장소 내려받기 — 디스크 때문에 한 번에 하나) · big(GPU를 크게 쓴다 — 한 번에 하나)
# 이미 다른 오케스트레이터로 돌고 있던 작업(res_1440000, 정확도 기준선 fp16, B kd_only)은 같은 결과 파일 이름으로 두어 알아보게 한다.
PY=/workspace/venv_main/bin/python; PG=/workspace/venv_gemma/bin/python; O=eval_results/full799
FIX=checkpoints/baseline_v2_fixa_native_tower_full799.pt
SIX="base_3b student_baseline_v2 student_kd_only_v4 student_temporal student_spatial student_full"
j(){ echo "$1|$2|$3|$4|$5|$6"; }
# --- 1단계: 비전 인코더 ---
# 원본 해상도도 모델별로 나눈다 (2026-10-21 컨테이너 재시작으로 6모델 묶음 실행을 잃은 뒤)
for MP in 1440000 802816 401408 200704 100352; do for m in $SIX; do
  G=14; [ $MP = 1440000 ] && G=32; [ $MP = 802816 ] && G=24; [ $MP = 401408 ] && G=18
  j "해상도 스윕 $MP $m" $O/res_${MP}__$m.json $G 1 gpu "$PY scripts/profile_vision_activations.py --fresh --log_every 0 --min_pixels 3136 --max_pixels $MP --only $m --out $O/res_${MP}__$m.json --dump $O/res_${MP}__${m}_per_image.jsonl"
done; done
j "결정성 점검" $O/res_1440000_rep2.json 32 1 gpu "$PY scripts/profile_vision_activations.py --fresh --log_every 0 --min_pixels 3136 --max_pixels 1440000 --only student_baseline_v2 --out $O/res_1440000_rep2.json --dump $O/res_1440000_rep2_per_image.jsonl"
j "v1 대 v2" $O/v1_vs_v2_profile.json 12 1 gpu "$PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ --extra student_baseline_v1=checkpoints/student_baseline/epoch_1 student_kd_only_v1=checkpoints/student_kd_only/epoch_1 --out $O/v1_vs_v2_profile.json --dump $O/v1_vs_v2_per_image.jsonl"
j "INT8" $O/int8_quant_error_native.json 32 1 gpu "$PY scripts/profile_int8_quant_error.py --max_pixels 1440000 --limit 799 --log_every 0 --out $O/int8_quant_error_native.json --per_image $O/int8_native_per_image.jsonl"
j "개입 fixA/fixB" $O/mlp_overflow.json 32 1 gpu "$PY scripts/probe_mlp_overflow.py --n_images 799 --out $O/mlp_overflow.json --per_image $O/mlp_overflow_per_image.jsonl"
j "가중치 편집 등가" $O/fixa_weight_equiv.json 32 1 gpu "$PY scripts/probe_fixa_weight_equiv.py --limit 799 --out $O/fixa_weight_equiv.json --per_image $O/fixa_weight_equiv_per_image.jsonl"
j "채널 849" $O/channel849.json 32 1 gpu "$PY scripts/probe_channel849.py --limit 799 --out $O/channel849.json --per_image $O/channel849_per_image.jsonl"
j "blk31 분해" $O/blk31_decompose.json 16 1 gpu "$PY scripts/probe_blk31_decompose.py --n_images 799 --out $O/blk31_decompose.json --per_image $O/blk31_decompose_per_image.jsonl"
j "잔차 개입" $O/blk31_intervention.json 16 1 gpu "$PY scripts/probe_blk31_intervention.py --n_images 799 --out $O/blk31_intervention.json --per_image $O/blk31_intervention_per_image.jsonl"
for OFF in 0 200 400 600; do N=200; [ $OFF -eq 600 ] && N=199
  j "norm2 방향 $OFF" $O/norm2_direction_off$OFF.json 16 1 gpu "$PY scripts/probe_norm2_direction.py --n_images $N --offset $OFF --out $O/norm2_direction_off$OFF.json --per_image $O/norm2_direction_off${OFF}_per_image.jsonl"
done
j "정렬 대상" $O/align_target_profile.json 12 1 gpu "$PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ --extra aligntgt_base3b=checkpoints/aligntgt_base3b/step_5000 --out $O/align_target_profile.json --dump $O/align_target_per_image.jsonl"
for L in 0.0 0.5 1.0 2.0; do
  j "λ 게이트 $L" $O/guard_sweep_align$L.json 32 1 gpu "$PY scripts/headroom_guard.py diagnose --checkpoint checkpoints/sweep_align$L/step_5000 --dtype float16 --max_pixels 1440000 --limit 799 --out $O/guard_sweep_align$L.json --per_image $O/guard_sweep_align${L}_per_image.jsonl"
done
for m in baseline_v2 kd_only_v4 temporal spatial full; do EX=""; for s in 5000 10000 15000 20000 25000; do EX="$EX student_${m}@${s}=checkpoints/student_${m}/step_${s}"; done
  j "학습 궤적 $m" $O/training_trajectory__$m.json 12 1 gpu "$PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ --extra$EX --out $O/training_trajectory__$m.json --dump $O/training_trajectory__${m}_per_image.jsonl"
done
for f in 1.0 0.85 0.72 0.62 0.55 0.48 0.42 0.36 0.3 0.24 0.18 0.92 0.88 0.82 0.78 0.75 0.69 0.66; do
  j "합성 연속체 ×$f" $O/detection_curve__f$f.json 32 1 gpu "$PY scripts/detection_curve.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --limit 799 --factors $f --out $O/detection_curve__f$f.json --fig $O/detection_curve__f$f.png --per_image $O/detection_curve__f${f}_per_image.jsonl"
done
# --- 2단계: 정확도 11,309문항 ---
NAT="--max_pixels 1440000 --min_pixels 3136"
# 정확도는 8조각으로 나눈다(조각당 약 1,414문항) — 컨테이너가 다시 켜져도 잃는 것이 한 조각 안이 되도록 (2026-10-21)
NSH=8
ev(){ for k in $(seq 0 $((NSH-1))); do j "정확도 $1 $2 조각 $k/$NSH" $O/${1}_epoch_1_${2}_shard${k}of${NSH}_nuscenesqa_val.json 20 1 gpu "$PY scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/$1/epoch_1 --dump_predictions --log_every 1000 --out_dir $O $3 --tag $2 --shard $k --num_shards $NSH"; done; }
ev student_baseline_v2 native_fp16      "$NAT --lora_dtype float16"
ev student_baseline_v2 native_fp16_fixa "$NAT --lora_dtype float16 --vision_state $FIX"
ev student_baseline_v2 native_bf16      "$NAT --lora_dtype bfloat16"
ev student_full        native_fp16      "$NAT --lora_dtype float16"
ev student_full        native_bf16      "$NAT --lora_dtype bfloat16"
ev student_baseline_v2 eval_fp16        "--lora_dtype float16"
ev student_baseline_v2 eval_bf16        "--lora_dtype bfloat16"
ev student_full        eval_fp16        "--lora_dtype float16"
ev student_full        eval_bf16        "--lora_dtype bfloat16"
# --- 3단계: 범위 확장 A·B, 도구 검증, 다른 계열, CAM_BACK ---
for m in pretrained baseline_v2 kd_only_v4 temporal spatial full; do ck=checkpoints/student_$m/epoch_1; [ $m = pretrained ] && ck=none
  j "B $m" $O/llm_headroom_$m.json 16 1 gpu "$PY scripts/probe_llm_headroom.py --checkpoint $ck --tag $m --limit 799 --out_dir $O --per_image $O/llm_headroom_${m}_per_image.jsonl"
done
for m in gemma-4-26B-A4B-it gemma-4-31B-it; do for st in 280 1120; do
  j "A $m st$st" $O/gemma4_vision_${m}_st${st}.json 16 1 gpu "$PG scripts/probe_gemma4_vision.py --model_dir /workspace/gemma4_vision/$m --max_soft_tokens $st --limit 799 --out_dir $O --per_image $O/gemma4_vision_${m}_st${st}_per_image.jsonl"
done; done
j "A 종단 799" $O/gemma4_e2e_26B-A4B_full799.json 60 1 big "$PG scripts/probe_gemma4_e2e.py --offset 0 --limit 799 --out $O/gemma4_e2e_26B-A4B_full799.json"
G="--dtype float16 --limit 799"
j "도구 기준선·DLA" $O/guard_dla_demo.json 32 1 gpu "$PY scripts/headroom_guard.py diagnose --checkpoint checkpoints/student_baseline_v2/epoch_1 $G --max_pixels 1440000 --available_formats float16,int8 --out $O/guard_dla_demo.json --per_image $O/guard_dla_demo_per_image.jsonl"
j "도구 사전학습 3B fix" $O/headroom_guard_fix_base3b.json 32 1 gpu "$PY scripts/headroom_guard.py fix --checkpoint none $G --max_pixels 1440000 --target_headroom 2.0 --verify --out $O/headroom_guard_fix_base3b.json --per_image $O/headroom_guard_fix_base3b_per_image.jsonl"
j "CAM_BACK 검증 799" $O/altimages_CAM_BACK_val799.json 40 1 big "$PY scripts/profile_alt_images.py --camera CAM_BACK --val_only --limit 799 --min_pixels 3136 --max_pixels 1440000 --only $SIX teacher_7b --out $O/altimages_CAM_BACK_val799.json --per_image $O/altimages_CAM_BACK_val799_per_image.jsonl"
# 공개 저장소 — 내려받아 재고 지운다. 디스크 때문에 한 번에 하나(dl)
j "도구 Qwen2-VL-2B-AWQ" $O/guard_qwen2vl_2b_awq.json 20 15 dl "$PY scripts/headroom_guard.py diagnose --repo Qwen/Qwen2-VL-2B-Instruct-AWQ $G --max_pixels 1440000 --out $O/guard_qwen2vl_2b_awq.json --per_image $O/guard_qwen2vl_2b_awq_per_image.jsonl"
j "도구 Qwen2-VL-7B-AWQ" $O/guard_qwen2vl_7b_awq.json 24 20 dl "$PY scripts/headroom_guard.py diagnose --repo Qwen/Qwen2-VL-7B-Instruct-AWQ $G --max_pixels 1440000 --out $O/guard_qwen2vl_7b_awq.json --per_image $O/guard_qwen2vl_7b_awq_per_image.jsonl"
j "도구 SmolVLM" $O/headroom_guard_smolvlm.json 20 15 dl "$PY scripts/headroom_guard.py diagnose --repo HuggingFaceTB/SmolVLM-Instruct $G --out $O/headroom_guard_smolvlm.json --per_image $O/headroom_guard_smolvlm_per_image.jsonl"
j "자연 상실" $O/natural_collapse.json 24 25 dl "$PY scripts/probe_natural_collapse.py --limit 799 --max_pixels 1440000 --cleanup --out $O/natural_collapse.json --per_image $O/natural_collapse_per_image.jsonl"
j "다른 계열 프로파일" $O/other_vlm_families_profile.json 24 25 dl "$PY scripts/profile_other_vlm_families.py --limit 799 --max_pixels 1440000 --cleanup --out $O/other_vlm_families_profile.json --per_image $O/other_vlm_families_per_image.jsonl"
j "다른 계열 INT8" $O/int8_across_families.json 24 25 dl "$PY scripts/probe_int8_across_families.py --limit 799 --max_pixels 1440000 --cleanup --out $O/int8_across_families.json --per_image $O/int8_across_families_per_image.jsonl"
j "직접 AngelSlim" $O/vision_tower_direct.json 24 25 dl "$PY scripts/probe_vision_tower_direct.py --repos AngelSlim/Qwen2.5-VL-3B-Instruct-INT4-AWQ AngelSlim/Qwen2.5-VL-7B-Instruct-AWQ --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct.json --per_image $O/vision_tower_direct_per_image.jsonl"
j "직접 공식" $O/vision_tower_direct_official.json 24 25 dl "$PY scripts/probe_vision_tower_direct.py --repos Qwen/Qwen2.5-VL-3B-Instruct-AWQ Qwen/Qwen2.5-VL-7B-Instruct-AWQ Qwen/Qwen2.5-VL-3B-Instruct --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct_official.json --per_image $O/vision_tower_direct_official_per_image.jsonl"
j "직접 상속 쌍" $O/vision_tower_inheritance.json 24 30 dl "$PY scripts/probe_vision_tower_direct.py --repos llava-hf/llava-1.5-7b-hf ybelkada/llava-1.5-7b-hf-awq microsoft/Phi-3.5-vision-instruct Isotr0py/Phi-3.5-vision-instruct-AWQ sensenova/SenseNova-SI-InternVL3-8B ELVISIO/SenseNova-SI-InternVL3-8B-AWQ --trust_remote_code --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_inheritance.json --per_image $O/vision_tower_inheritance_per_image.jsonl"
j "직접 Pixtral" $O/vision_tower_direct_pixtral.json 32 30 dl "$PY scripts/probe_vision_tower_direct.py --repos mistral-community/pixtral-12b --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct_pixtral.json --per_image $O/vision_tower_direct_pixtral_per_image.jsonl"
j "직접 Kimi" $O/vision_tower_direct_kimi.json 40 40 dl "$PY scripts/probe_vision_tower_direct.py --repos moonshotai/Kimi-VL-A3B-Instruct --trust_remote_code --limit 799 --max_pixels 1440000 --cleanup --out $O/vision_tower_direct_kimi.json --per_image $O/vision_tower_direct_kimi_per_image.jsonl"
j "다리 측정(새 transformers)" $O/bridge_newhf.json 40 40 dl "$PY scripts/make_image_paths.py --root /workspace/vlm_project_2 --n 799 --out $O/bridge_image_paths.json && $PG scripts/bridge_measure_newhf.py --repos Qwen/Qwen2.5-VL-3B-Instruct QuantTrio/Qwen3-VL-32B-Instruct-AWQ QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ zai-org/GLM-4.1V-9B-Thinking moonshotai/Kimi-VL-A3B-Instruct --paths $O/bridge_image_paths.json --max_pixels 1440000 --cleanup --out $O/bridge_newhf.json --per_image $O/bridge_newhf_per_image.jsonl"
