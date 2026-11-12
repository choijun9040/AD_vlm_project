#!/bin/bash
# 개정 35~37 — A100이 켜지면 바뀐 스크립트·사슬을 올리고 서버 감시기를 한 번 돌려 rev35 사슬을 시작한다(2026-11-09). 한 번 성공하면 끝난다.
cd /home/cj/workspace/vlm_project_2
L=logs/upload_rev35.log
F="scripts/fidelity_rwqa.py scripts/flux_rev35.py scripts/test_auto_resid_towers.py scripts/test_auto_resid_vae.py scripts/test_auto_resid_flux.py scripts/sdxl_coco_equiv.py scripts/cmp_sdxl_lpips_probe.py"
until ssh -o ConnectTimeout=15 -o BatchMode=yes vast 'nvidia-smi >/dev/null' 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') 접속됨 — 올리기 시작" >> $L
ok=1
scp -q $F vast:/workspace/vlm_project_2/scripts/ >> $L 2>&1 || ok=0
scp -q scripts/chains/rev35.sh scripts/chains/supervise.sh vast:/workspace/vlm_project_2/scripts/chains/ >> $L 2>&1 || ok=0
ssh -o ConnectTimeout=15 vast 'cd /workspace/vlm_project_2; for f in eval_results/cmp/sdxl_calib.latents.pt eval_results/cmp/flux_prompts_calib.pt eval_results/cmp/flux_prompts.pt eval_results/cmp/sdxl_enc_resid_plan_coco.json eval_results/cmp/sdxl_calib_coco.json eval_results/cmp/flux_calib.json data/drivelm_official/train100.jsonl data/drivelm_official/val799.jsonl; do [ -e $f ] || echo "없음 $f"; done; /workspace/venv_sd/bin/python -c "import torch,diffusers;print(\"venv_sd\",torch.__version__,diffusers.__version__,hasattr(torch.backends.cuda.matmul,\"allow_fp16_accumulation\"))"' >> $L 2>&1
[ $ok = 1 ] && ssh -o ConnectTimeout=15 vast 'cd /workspace/vlm_project_2; bash scripts/chains/supervise.sh' >> $L 2>&1
echo "$(date '+%F %T') 끝 (올리기 성공=$ok)" >> $L
