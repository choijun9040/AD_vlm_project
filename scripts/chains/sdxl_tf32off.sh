#!/bin/bash
# 개정 39 — SDXL을 TF32 끄고 다시(기준 = 순수 fp32). c는 개정 38(env2)의 측정용 값. 결과는 eval_results/cmp/env3/. 2026-11-10
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PS=/workspace/venv_sd/bin/python; O=eval_results/cmp; E2=$O/env2; E=$O/env3; mkdir -p $E
cp -n $E2/sdxl_gen_cmp.latents.pt $E/sdxl_gen_cmp.latents.pt
CG=$(python3 -c "import json; print(json.load(open('$E2/sdxl_calib.json'))['c'])")
CC=$(python3 -c "import json; print(json.load(open('$E2/sdxl_calib_coco.json'))['c'])")
step t_gen $E/sdxl_gen_cmp.json -- bash -c "$PS scripts/cmp_sdxl_gen.py --tsv data/PartiPrompts.tsv --c $CG --no_tf32 --out $E/sdxl_gen_cmp.json >> logs/env3.log 2>&1" || exit 1
D=/dev/shm/coco; mkdir -p $D
[ -d $D/val2017 ] && [ "$(ls $D/val2017 | wc -l)" = 5000 ] || rm -f logs/steps/t_cocodl.done
step t_cocodl -- bash -c "cd $D && curl -sSL --retry 5 -C - -o val2017.zip https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip && unzip -tq val2017.zip && unzip -q -o val2017.zip && rm -f val2017.zip && [ \$(ls val2017 | wc -l) = 5000 ]" || exit 1
step t_coco $E/sdxl_coco.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco $D/val2017 --no_tf32 --lpips_input both --enc_plan $E2/sdxl_enc_resid_plan_coco.json --dec_c $CC --out $E/sdxl_coco.json >> logs/env3.log 2>&1" || exit 1
step t_genc $E/sdxl_coco_genc.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco $D/val2017 --no_tf32 --lpips_input both --enc_plan $E2/sdxl_enc_resid_plan.json --dec_c $CG --rows resid_dec_fp16_enc_fp32,resid_full_fp16 --out $E/sdxl_coco_genc.json >> logs/env3.log 2>&1" || exit 1
touch logs/env3.done; log "사슬 끝"
