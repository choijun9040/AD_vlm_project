#!/bin/bash
# 개정 35 — 공식 Qwen·Gemma (R) fp32 등가성 799장 전체, FLUX (R) fp32 등가성 100개 전체, Draw Things 엔진 정밀도 재현(b2). 2026-11-09
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PY=/workspace/venv_main/bin/python; PG=/workspace/venv_gemma/bin/python; PS=/workspace/venv_sd/bin/python
O=eval_results/cmp; V=data/drivelm_official/val799.jsonl; C=data/drivelm_official/train100.jsonl
step e_qwen $O/dlm_equiv_qwen.json -- bash -c "$PY scripts/fidelity_rwqa.py qwen --forms fp32_resid --q $V --calib_q $C --out $O/dlm_equiv_qwen.json >> logs/rev35_equiv_qwen.log 2>&1" || exit 1
for m in 26B-A4B 31B; do for t in 280 1120; do
  D=/workspace/gemma4_vision/gemma-4-$m-it
  step e_gemma_${m}_$t $O/dlm_equiv_gemma_${m}_$t.json -- bash -c "$PG scripts/fidelity_rwqa.py gemma --model_dir $D --tokens $t --forms fp32_resid --q $V --calib_q $C --out $O/dlm_equiv_gemma_${m}_$t.json >> logs/rev35_equiv_gemma.log 2>&1" || exit 1
done; done
# 개정 36 — 자동 구성 도구를 측정용 100개로(진단·c·등가성·수동 구성 대조)
step a_qwen $O/auto100_qwen.json -- bash -c "$PY scripts/test_auto_resid_towers.py qwen --calib $C --out $O/auto100_qwen.json >> logs/rev36_auto.log 2>&1" || exit 1
step a_gemma $O/auto100_gemma.json -- bash -c "$PG scripts/test_auto_resid_towers.py gemma --calib $C --out $O/auto100_gemma.json >> logs/rev36_auto.log 2>&1" || exit 1
step a_vae $O/auto100_vae.json -- bash -c "$PS scripts/test_auto_resid_vae.py --latents $O/sdxl_calib.latents.pt --out $O/auto100_vae.json >> logs/rev36_auto.log 2>&1" || exit 1
step a_flux $O/auto100_flux.json -- bash -c "$PS scripts/test_auto_resid_flux.py --emb $O/flux_prompts_calib.pt --n 100 --out $O/auto100_flux.json >> logs/rev36_auto.log 2>&1" || exit 1
# 개정 36 규칙: 도구 구성이 수동 구성과 다르면(매개변수·eps) 그 대상의 결과를 도구 구성으로 다시 측정한다 — 다름을 표시하고 알린다
python3 -c "
import json,sys
bad=[]
for t in ('qwen','gemma','vae','flux'):
    d=json.load(open('$O/auto100_'+t+'.json'))
    if not d.get('ok') or not d.get('same_params') or not d.get('same_eps'): bad.append(t)
print(' '.join(bad)); sys.exit(1 if bad else 0)" > logs/rev36_mismatch.txt 2>&1 && log "개정 36: 네 대상 모두 수동 구성과 같다" || { log "**개정 36: 구성 다름 또는 실패 — $(cat logs/rev36_mismatch.txt) — 도구 구성으로 재측정 필요(규칙)**"; touch logs/rev36_mismatch.flag; }
# 개정 37 — 부분 표본 값을 평가 전체로: ① LPIPS 설정 시험 5,000장 ②③ COCO 인코더·디코더 등가성 5,000장 ④ Qwen 같은 228장 비교
D=/dev/shm/coco; mkdir -p $D
[ -d $D/val2017 ] && [ "$(ls $D/val2017 | wc -l)" = 5000 ] || rm -f logs/steps/s_coco.done
step s_coco -- bash -c "cd $D && curl -sSL --retry 5 -C - -o val2017.zip https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip && unzip -tq val2017.zip && unzip -q -o val2017.zip && rm -f val2017.zip && [ \$(ls val2017 | wc -l) = 5000 ]" || exit 1
step l_probe5k $O/sdxl_lpips_probe5000.json -- bash -c "$PS scripts/cmp_sdxl_lpips_probe.py --coco $D/val2017 --n 5000 --out $O/sdxl_lpips_probe5000.json >> logs/rev37.log 2>&1" || exit 1
step c_equiv $O/sdxl_coco_equiv.json -- bash -c "$PS scripts/sdxl_coco_equiv.py --coco $D/val2017 --out $O/sdxl_coco_equiv.json >> logs/rev37.log 2>&1" || exit 1
step q_perimg $O/dlm_fid_qwen_perimg.json -- bash -c "$PY scripts/fidelity_rwqa.py qwen --forms bf16,fp16,fp16_resid --per_image --q $V --calib_q $C --out $O/dlm_fid_qwen_perimg.json >> logs/rev37.log 2>&1" || exit 1
step f_check -- bash -c "$PS scripts/flux_rev35.py --emb $O/flux_prompts.pt --out $O/flux_rev35.json --check >> logs/rev35_flux.log 2>&1" || exit 1
step f_run $O/flux_rev35.json -- bash -c "$PS scripts/flux_rev35.py --emb $O/flux_prompts.pt --out $O/flux_rev35.json >> logs/rev35_flux.log 2>&1" || exit 1
touch logs/rev35.done; log "사슬 끝"
