#!/bin/bash
# 개정 34 — LPIPS 원문 값 재현 시험(COCO val2017 앞 1,000장, 원래 VAE·FP16-Fix fp32, alex·vgg × [-1,1]·[0,1]). 2026-11-06
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PS=/workspace/venv_sd/bin/python; D=/dev/shm/coco; mkdir -p $D
[ -d $D/val2017 ] && [ "$(ls $D/val2017 | wc -l)" = 5000 ] || rm -f logs/steps/s_coco.done
step s_coco -- bash -c "cd $D && curl -sSL --retry 5 -C - -o val2017.zip https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip && unzip -tq val2017.zip && unzip -q -o val2017.zip && rm -f val2017.zip && [ \$(ls val2017 | wc -l) = 5000 ]" || exit 1
step l_probe eval_results/cmp/sdxl_lpips_probe.json -- bash -c "$PS scripts/cmp_sdxl_lpips_probe.py --coco $D/val2017 --n 1000 --out eval_results/cmp/sdxl_lpips_probe.json >> logs/sdxl_lpips.log 2>&1" || exit 1
# 재현 설정(alex·[0,1])이 하나였다 → 규칙대로 COCO 5,000장 표를 그 설정으로(c는 COCO train2017로 정한 값 — 개정 33)
O=eval_results/cmp; DC=$(python3 -c "import json; print(json.load(open('$O/sdxl_calib_coco.json'))['c'])")
step l_full $O/sdxl_coco_lp01.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco $D/val2017 --lpips_input 01 --enc_plan $O/sdxl_enc_resid_plan_coco.json --dec_c $DC --rows fix_fp32,fix_fp16,resid_dec_fp16_enc_fp32,resid_full_fp16 --out $O/sdxl_coco_lp01.json >> logs/sdxl_lpips.log 2>&1" || exit 1
touch logs/rev34_lpips.done; log "사슬 끝"
