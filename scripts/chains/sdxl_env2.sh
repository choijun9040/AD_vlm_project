#!/bin/bash
# 개정 38 — SDXL 결과를 새 환경(vast2)에서 다시. 입력 잠재는 옛 환경에서 만든 같은 파일. 결과는 eval_results/cmp/env2/. 2026-11-09
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PS=/workspace/venv_sd/bin/python; O=eval_results/cmp; E=$O/env2; mkdir -p $E data
cp -n $O/sdxl_calib.latents.pt $E/sdxl_calib.latents.pt; cp -n $O/sdxl_gen_cmp.latents.pt $E/sdxl_gen_cmp.latents.pt
step e2_tsv data/PartiPrompts.tsv -- bash -c "curl -sSL --retry 5 -o data/PartiPrompts.tsv https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv && [ \$(wc -l < data/PartiPrompts.tsv) -gt 1600 ]" || exit 1
step e2_gcalib $E/sdxl_calib.json -- bash -c "$PS scripts/cmp_sdxl_calib.py --tsv data/PartiPrompts.tsv --out $E/sdxl_calib.json >> logs/env2.log 2>&1" || exit 1
CG=$(python3 -c "import json; print(json.load(open('$E/sdxl_calib.json'))['c'])")
step e2_gen $E/sdxl_gen_cmp.json -- bash -c "$PS scripts/cmp_sdxl_gen.py --tsv data/PartiPrompts.tsv --c $CG --out $E/sdxl_gen_cmp.json >> logs/env2.log 2>&1" || exit 1
step e2_view $E/sdxl_gen_view.png -- bash -c "$PS scripts/cmp_sdxl_view.py >> logs/env2.log 2>&1 && cp $O/sdxl_gen_view.png $E/sdxl_gen_view.png" || exit 1
T=/dev/shm/coco/train100; mkdir -p $T
[ "$(ls $T 2>/dev/null | grep -c jpg)" = 100 ] || rm -f logs/steps/e2_train.done
step e2_train -- bash -c "cd $T && curl -sSL --retry 5 'https://s3.amazonaws.com/images.cocodataset.org?list-type=2&prefix=train2017/&max-keys=100' | python3 -c \"import sys,re; print('\n'.join(re.findall(r'<Key>(train2017/[0-9]+\.jpg)</Key>', sys.stdin.read())))\" > keys.txt && [ \$(wc -l < keys.txt) = 100 ] && while read k; do curl -sSL --retry 5 -o \$(basename \$k) https://s3.amazonaws.com/images.cocodataset.org/\$k || exit 1; done < keys.txt && rm keys.txt && [ \$(ls *.jpg | wc -l) = 100 ]" || exit 1
step e2_enc_coco $E/sdxl_enc_resid_plan_coco.json -- bash -c "$PS scripts/sdxl_encoder_resid.py --images $T --out $E/sdxl_enc_resid_plan_coco.json >> logs/env2.log 2>&1" || exit 1
step e2_dec_coco $E/sdxl_calib_coco.json -- bash -c "$PS scripts/cmp_sdxl_calib_coco.py --images $T --out $E/sdxl_calib_coco.json >> logs/env2.log 2>&1" || exit 1
step e2_enc_gen $E/sdxl_enc_resid_plan.json -- bash -c "$PS scripts/sdxl_encoder_resid.py --latents $E/sdxl_calib.latents.pt --out $E/sdxl_enc_resid_plan.json >> logs/env2.log 2>&1" || exit 1
# COCO val — rev35 사슬의 s_coco는 2시간 뒤라 여기서 받는다. 받은 뒤 rev35의 s_coco 완료 표시를 남겨 같은 폴더를 덮어쓰지 않게 한다(2026-11-09)
D=/dev/shm/coco; mkdir -p $D
[ -d $D/val2017 ] && [ "$(ls $D/val2017 | wc -l)" = 5000 ] || rm -f logs/steps/e2_cocodl.done
step e2_cocodl -- bash -c "cd $D && curl -sSL --retry 5 -C - -o val2017.zip https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip && unzip -tq val2017.zip && unzip -q -o val2017.zip && rm -f val2017.zip && [ \$(ls val2017 | wc -l) = 5000 ]" || exit 1
touch logs/steps/s_coco.done
CC=$(python3 -c "import json; print(json.load(open('$E/sdxl_calib_coco.json'))['c'])")
step e2_coco $E/sdxl_coco.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco $D/val2017 --lpips_input both --enc_plan $E/sdxl_enc_resid_plan_coco.json --dec_c $CC --out $E/sdxl_coco.json >> logs/env2.log 2>&1" || exit 1
step e2_coco_genc $E/sdxl_coco_genc.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco $D/val2017 --lpips_input both --enc_plan $E/sdxl_enc_resid_plan.json --dec_c $CG --rows resid_dec_fp16_enc_fp32,resid_full_fp16 --out $E/sdxl_coco_genc.json >> logs/env2.log 2>&1" || exit 1
step e2_equiv $E/sdxl_coco_equiv.json -- bash -c "$PS scripts/sdxl_coco_equiv.py --coco $D/val2017 --enc_plan $E/sdxl_enc_resid_plan_coco.json --dec_calib $E/sdxl_calib_coco.json --out $E/sdxl_coco_equiv.json >> logs/env2.log 2>&1" || exit 1
touch logs/env2.done; log "사슬 끝"
