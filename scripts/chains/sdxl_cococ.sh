#!/bin/bash
# 개정 33 — COCO train2017 100장(S3 목록 첫 100개, 이미지 번호 순)으로 SDXL-VAE (R)의 c를 다시 정하고 COCO val2017 5,000장에 두 행. 2026-11-06
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PS=/workspace/venv_sd/bin/python; T=/dev/shm/coco/train100; O=eval_results/cmp
[ -d $T ] && [ "$(ls $T | wc -l)" = 100 ] || rm -f logs/steps/c_train.done
step c_train -- bash -c "mkdir -p $T && cd $T && curl -sSL --retry 5 'https://s3.amazonaws.com/images.cocodataset.org?list-type=2&prefix=train2017/&max-keys=100' \
  | python3 -c \"import sys,re; print('\n'.join(re.findall(r'<Key>(train2017/[0-9]+\.jpg)</Key>', sys.stdin.read())))\" > keys.txt && [ \$(wc -l < keys.txt) = 100 ] \
  && while read k; do curl -sSL --retry 5 -o \$(basename \$k) https://s3.amazonaws.com/images.cocodataset.org/\$k || exit 1; done < keys.txt \
  && rm keys.txt && [ \$(ls *.jpg | wc -l) = 100 ] && $PS -c \"from PIL import Image; import glob; [Image.open(f).verify() for f in glob.glob('*.jpg')]\"" || exit 1
step c_enc $O/sdxl_enc_resid_plan_coco.json -- bash -c "$PS scripts/sdxl_encoder_resid.py --images $T --out $O/sdxl_enc_resid_plan_coco.json >> logs/sdxl_cococ.log 2>&1" || exit 1
step c_dec $O/sdxl_calib_coco.json -- bash -c "$PS scripts/cmp_sdxl_calib_coco.py --images $T --out $O/sdxl_calib_coco.json >> logs/sdxl_cococ.log 2>&1" || exit 1
ENC="--enc_plan $O/sdxl_enc_resid_plan_coco.json"; ROWS=resid_dec_fp16_enc_fp32,resid_full_fp16
python3 -c "import json,sys; sys.exit(0 if json.load(open('$O/sdxl_enc_resid_plan_coco.json')).get('ok') else 1)" || { log "**COCO 인코더 (R) 구성 실패 또는 등가성 미달 — (R) 인코더 행은 쓰지 않는다(등록 규칙)**"; ENC=""; ROWS=resid_dec_fp16_enc_fp32; }
DC=$(python3 -c "import json; print(json.load(open('$O/sdxl_calib_coco.json'))['c'])")
step c_eval $O/sdxl_coco_cococ.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco /dev/shm/coco/val2017 $ENC --dec_c $DC --rows $ROWS --out $O/sdxl_coco_cococ.json >> logs/sdxl_cococ.log 2>&1" || exit 1
touch logs/rev33_cococ.done; log "사슬 끝"
