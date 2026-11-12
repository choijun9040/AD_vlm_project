#!/bin/bash
# COCO 공식 링크(images.cocodataset.org)는 HTTPS 인증서 이름이 맞지 않는다 — 같은 S3 버킷을 경로 방식 주소로(올바른 AWS 인증서) 받고 zip 무결성을 확인한다
# 개정 31 — SDXL을 FP16-Fix 원문 조건(COCO val2017 256×256, LPIPS·SSIM·MSE)으로 (이어 하기 판). 2026-11-06
set -u; cd /workspace/vlm_project_2; . scripts/chains/common.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HOME=/workspace/.hf_home
PS=/workspace/venv_sd/bin/python; D=/dev/shm/coco
mkdir -p $D
[ -d $D/val2017 ] && [ "$(ls $D/val2017 | wc -l)" = 5000 ] || rm -f logs/steps/s_coco.done
step s_coco -- bash -c "cd $D && curl -sSL --retry 5 -C - -o val2017.zip https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip && unzip -tq val2017.zip && unzip -q -o val2017.zip && rm -f val2017.zip && [ \$(ls val2017 | wc -l) = 5000 ]" || exit 1
step s_deps -- bash -c "$PS -c 'import lpips, skimage' 2>/dev/null || /workspace/venv_sd/bin/pip install -q lpips scikit-image" || exit 1
step s_eval eval_results/cmp/sdxl_coco.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco $D/val2017 --out eval_results/cmp/sdxl_coco.json >> logs/sdxl_coco.log 2>&1" || exit 1
# 개정 32 — 인코더 (R): 측정용 생성 이미지로 자동 구성 → COCO 원문 조건에 「(R) 인코더 + (R) 디코더, 모두 fp16」 행
step s_encplan eval_results/cmp/sdxl_enc_resid_plan.json -- bash -c "$PS scripts/sdxl_encoder_resid.py --latents eval_results/cmp/sdxl_calib.latents.pt --out eval_results/cmp/sdxl_enc_resid_plan.json >> logs/sdxl_encplan.log 2>&1" || exit 1
python3 -c "import json,sys; sys.exit(0 if json.load(open('eval_results/cmp/sdxl_enc_resid_plan.json')).get('ok') else 1)" || { log "**인코더 (R) 구성 실패 또는 등가성 미달 — 행을 더하지 않는다(등록 규칙)**"; touch logs/rev31_sdxl.done; exit 0; }
step s_eval_full eval_results/cmp/sdxl_coco_full.json -- bash -c "$PS scripts/cmp_sdxl_coco.py --coco $D/val2017 --enc_plan eval_results/cmp/sdxl_enc_resid_plan.json --rows resid_full_fp16 --out eval_results/cmp/sdxl_coco_full.json >> logs/sdxl_coco_full.log 2>&1" || exit 1
touch logs/rev31_sdxl.done; log "사슬 끝"
