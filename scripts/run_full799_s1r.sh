#!/bin/bash
# 평가 데이터 전부로 통일 — 1단계(이어 하기 판): 비전 인코더 핵심 사슬 (사전 등록 4fddeb4)
# 내용은 run_full799_s1.sh와 같다. 단계마다 완료 표시·재시도·자원 대기를 둔다(full799_lib.sh).
set -u; source /workspace/vlm_project_2/scripts/full799_lib.sh
until [ -f checkpoints/baseline_v2_fixa_native_tower_full799.pt ] && ! pgrep -f "headroom_guard.py fix --checkpoint checkpoints/student_baseline_v2" >/dev/null; do sleep 60; done
SIX="base_3b student_baseline_v2 student_kd_only_v4 student_temporal student_spatial student_full"
for MP in 1440000 802816 401408 200704 100352; do
  step "해상도 스윕 $MP" $O/res_$MP.json 16 5 $PY scripts/profile_vision_activations.py --fresh --log_every 0 --min_pixels 3136 --max_pixels $MP \
      --only $SIX --out $O/res_$MP.json --dump $O/res_${MP}_per_image.jsonl
done
step "결정성 점검" $O/res_1440000_rep2.json 16 2 $PY scripts/profile_vision_activations.py --fresh --log_every 0 --min_pixels 3136 --max_pixels 1440000 \
    --only student_baseline_v2 --out $O/res_1440000_rep2.json --dump $O/res_1440000_rep2_per_image.jsonl
step "v1 대 v2" $O/v1_vs_v2_profile.json 12 2 $PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ \
    --extra student_baseline_v1=checkpoints/student_baseline/epoch_1 student_kd_only_v1=checkpoints/student_kd_only/epoch_1 \
    --out $O/v1_vs_v2_profile.json --dump $O/v1_vs_v2_per_image.jsonl
step "INT8" $O/int8_quant_error_native.json 16 2 $PY scripts/profile_int8_quant_error.py --max_pixels 1440000 --limit 799 --log_every 0 \
    --out $O/int8_quant_error_native.json --per_image $O/int8_native_per_image.jsonl
step "개입 fixA/fixB" $O/mlp_overflow.json 16 2 $PY scripts/probe_mlp_overflow.py --n_images 799 --out $O/mlp_overflow.json --per_image $O/mlp_overflow_per_image.jsonl
step "가중치 편집 등가" $O/fixa_weight_equiv.json 16 2 $PY scripts/probe_fixa_weight_equiv.py --limit 799 --out $O/fixa_weight_equiv.json --per_image $O/fixa_weight_equiv_per_image.jsonl
step "채널 849" $O/channel849.json 16 2 $PY scripts/probe_channel849.py --limit 799 --out $O/channel849.json --per_image $O/channel849_per_image.jsonl
step "blk31 분해" $O/blk31_decompose.json 16 2 $PY scripts/probe_blk31_decompose.py --n_images 799 --out $O/blk31_decompose.json --per_image $O/blk31_decompose_per_image.jsonl
step "잔차 개입" $O/blk31_intervention.json 16 2 $PY scripts/probe_blk31_intervention.py --n_images 799 --out $O/blk31_intervention.json --per_image $O/blk31_intervention_per_image.jsonl
for OFF in 0 200 400 600; do
  N=200; [ $OFF -eq 600 ] && N=199
  step "norm2 방향 $OFF" $O/norm2_direction_off$OFF.json 16 2 $PY scripts/probe_norm2_direction.py --n_images $N --offset $OFF \
      --out $O/norm2_direction_off$OFF.json --per_image $O/norm2_direction_off${OFF}_per_image.jsonl
done
step "정렬 대상" $O/align_target_profile.json 12 2 $PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ \
    --extra aligntgt_base3b=checkpoints/aligntgt_base3b/step_5000 --out $O/align_target_profile.json --dump $O/align_target_per_image.jsonl
for L in 0.0 0.5 1.0 2.0; do
  step "λ 게이트 $L" $O/guard_sweep_align$L.json 24 2 $PY scripts/headroom_guard.py diagnose --checkpoint checkpoints/sweep_align$L/step_5000 --dtype float16 \
      --max_pixels 1440000 --limit 799 --out $O/guard_sweep_align$L.json --per_image $O/guard_sweep_align${L}_per_image.jsonl
done
EXTRA=""; for m in baseline_v2 kd_only_v4 temporal spatial full; do for s in 5000 10000 15000 20000 25000; do
  EXTRA="$EXTRA student_${m}@${s}=checkpoints/student_${m}/step_${s}"; done; done
step "학습 궤적" $O/training_trajectory_profile.json 12 3 $PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ --extra $EXTRA \
    --out $O/training_trajectory_profile.json --dump $O/training_trajectory_per_image.jsonl
step "합성 연속체 거친" $O/detection_curve.json 24 2 $PY scripts/detection_curve.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --limit 799 \
    --factors 1.0,0.85,0.72,0.62,0.55,0.48,0.42,0.36,0.3,0.24,0.18 --out $O/detection_curve.json --fig $O/detection_curve.png --per_image $O/detection_curve_per_image.jsonl
step "합성 연속체 촘촘" $O/detection_curve_fine.json 24 2 $PY scripts/detection_curve.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --limit 799 \
    --factors 0.92,0.88,0.85,0.82,0.78,0.75,0.72,0.69,0.66 --out $O/detection_curve_fine.json --fig $O/detection_curve_fine.png --per_image $O/detection_curve_fine_per_image.jsonl
stage_done s1r
