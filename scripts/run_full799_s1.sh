#!/bin/bash
# 평가 데이터 전부로 통일 — 1단계: 비전 인코더 핵심 사슬 (사전 등록 4fddeb4)
# 모두 DriveLM 검증 CAM_FRONT 799장(149장면), 결과는 eval_results/full799/ 새 파일로, 이미지별 기록을 남긴다.
set -u
cd /workspace/vlm_project_2; export HF_HOME=/workspace/hf_home
PY=/workspace/venv_main/bin/python; O=eval_results/full799; mkdir -p $O
log(){ echo "[$(date '+%F %T')] $*"; }
run(){ log "시작 $1"; shift; "$@" 2>&1 | grep -E "^\[|저장|→|이미지별|Error|Traceback|error|상실|여유" | tail -25; log "끝 (종료 ${PIPESTATUS[0]})"; }

# 다른 GPU 작업(A1, 799장 교정 풀기)이 끝날 때까지 기다린다 — 시간 측정이 섞이지 않게
while pgrep -f "probe_gemma4_vision.py|headroom_guard.py fix" >/dev/null; do sleep 60; done
# 필요한 체크포인트가 전송될 때까지 기다린다
for f in checkpoints/student_full/step_25000 checkpoints/sweep_align2.0/step_5000 checkpoints/aligntgt_base3b/step_5000 checkpoints/student_kd_only/epoch_1; do
  until [ -f $f/adapter_model.safetensors ]; do sleep 60; done; done
sleep 120   # 마지막 파일 쓰기가 끝나도록
SIX="base_3b student_baseline_v2 student_kd_only_v4 student_temporal student_spatial student_full"

# ① 해상도 스윕 — 여유·상실 (Ⅳ장 [res-collapse]·[inversion]·[dtype-rule], Ⅶ장 [exceed])
for MP in 1440000 802816 401408 200704 100352; do
  run "해상도 스윕 $MP" $PY scripts/profile_vision_activations.py --fresh --log_every 0 --min_pixels 3136 --max_pixels $MP \
      --only $SIX --out $O/res_$MP.json --dump $O/res_${MP}_per_image.jsonl
done
# ② 결정성 점검 — 같은 스크립트·같은 799장으로 기준선 원본 fp16을 한 번 더
run "결정성 점검" $PY scripts/profile_vision_activations.py --fresh --log_every 0 --min_pixels 3136 --max_pixels 1440000 \
    --only student_baseline_v2 --out $O/res_1440000_rep2.json --dump $O/res_1440000_rep2_per_image.jsonl
# ③ v1 대 v2 (평가 해상도, 기본 설정)
run "v1 대 v2" $PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ \
    --extra student_baseline_v1=checkpoints/student_baseline/epoch_1 student_kd_only_v1=checkpoints/student_kd_only/epoch_1 \
    --out $O/v1_vs_v2_profile.json --dump $O/v1_vs_v2_per_image.jsonl
# ④ INT8 양자화 오차 (원본 해상도)
run "INT8" $PY scripts/profile_int8_quant_error.py --max_pixels 1440000 --limit 799 --log_every 0 \
    --out $O/int8_quant_error_native.json --per_image $O/int8_native_per_image.jsonl
# ⑤ 기전 프로브
run "개입 fixA/fixB" $PY scripts/probe_mlp_overflow.py --n_images 799 --out $O/mlp_overflow.json --per_image $O/mlp_overflow_per_image.jsonl
run "가중치 편집 등가" $PY scripts/probe_fixa_weight_equiv.py --limit 799 --out $O/fixa_weight_equiv.json --per_image $O/fixa_weight_equiv_per_image.jsonl
run "채널 849" $PY scripts/probe_channel849.py --limit 799 --out $O/channel849.json --per_image $O/channel849_per_image.jsonl
run "blk31 분해" $PY scripts/probe_blk31_decompose.py --n_images 799 --out $O/blk31_decompose.json --per_image $O/blk31_decompose_per_image.jsonl
run "잔차 개입" $PY scripts/probe_blk31_intervention.py --n_images 799 --out $O/blk31_intervention.json --per_image $O/blk31_intervention_per_image.jsonl
# norm2는 이미지마다 텐서를 들고 있어 799장을 한 번에 돌리면 메모리가 넘친다 — 200장씩 나눠 잰다
for OFF in 0 200 400 600; do
  N=200; [ $OFF -eq 600 ] && N=199
  run "norm2 방향 $OFF" $PY scripts/probe_norm2_direction.py --n_images $N --offset $OFF --out $O/norm2_direction_off$OFF.json --per_image $O/norm2_direction_off${OFF}_per_image.jsonl
done
# ⑥ 정렬 대상 (평가 해상도; λ 799장 덤프와 짝)
run "정렬 대상" $PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ \
    --extra aligntgt_base3b=checkpoints/aligntgt_base3b/step_5000 --out $O/align_target_profile.json --dump $O/align_target_per_image.jsonl
# ⑦ λ 게이트 (원본 해상도, 도구)
for L in 0.0 0.5 1.0 2.0; do
  run "λ 게이트 $L" $PY scripts/headroom_guard.py diagnose --checkpoint checkpoints/sweep_align$L/step_5000 --dtype float16 \
      --max_pixels 1440000 --limit 799 --out $O/guard_sweep_align$L.json --per_image $O/guard_sweep_align${L}_per_image.jsonl
done
# ⑧ 학습 궤적 (평가 해상도, 25개 체크포인트)
EXTRA=""; for m in baseline_v2 kd_only_v4 temporal spatial full; do for s in 5000 10000 15000 20000 25000; do
  EXTRA="$EXTRA student_${m}@${s}=checkpoints/student_${m}/step_${s}"; done; done
run "학습 궤적" $PY scripts/profile_vision_activations.py --fresh --log_every 0 --only __none__ --extra $EXTRA \
    --out $O/training_trajectory_profile.json --dump $O/training_trajectory_per_image.jsonl
# ⑨ 합성 연속체 (모델 기본 해상도)
run "합성 연속체 거친" $PY scripts/detection_curve.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --limit 799 \
    --factors 1.0,0.85,0.72,0.62,0.55,0.48,0.42,0.36,0.3,0.24,0.18 --out $O/detection_curve.json --fig $O/detection_curve.png --per_image $O/detection_curve_per_image.jsonl
run "합성 연속체 촘촘" $PY scripts/detection_curve.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --limit 799 \
    --factors 0.92,0.88,0.85,0.82,0.78,0.75,0.72,0.69,0.66 --out $O/detection_curve_fine.json --fig $O/detection_curve_fine.png --per_image $O/detection_curve_fine_per_image.jsonl
log "1단계 완료"
