"""SDXL 생성 잠재 앞 8개를 원래 fp32·원래 fp16·FP16-Fix·(R) c=1/64로 디코딩해 격자 그림으로 저장(눈으로 확인용, 2026-11-04)."""
import copy, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_sdxl_vae_r import apply_resid_vae
from diffusers import AutoencoderKL
lat = torch.load("eval_results/cmp/sdxl_gen_cmp.latents.pt")[:8]
v32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
r = copy.deepcopy(v32); apply_resid_vae(r, 1 / 64)
arms = [("orig_fp32", v32, torch.float32), ("orig_fp16", copy.deepcopy(v32).half(), torch.float16),
        ("fix_fp16", AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda").eval(), torch.float16),
        ("resid_fp16", r.half(), torch.float16)]
rows = []
with torch.no_grad():
    for name, m, dt in arms:
        row = []
        for z in lat:
            x = m.decode(z.cuda().to(dt)).sample.float()
            x = torch.nan_to_num(x, nan=-1.0, posinf=1.0, neginf=-1.0)     # 비유한은 검정으로 보이게
            im = ((x[0] + 1) / 2).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            row.append(Image.fromarray((im * 255).round().astype(np.uint8)).resize((256, 256), Image.BICUBIC))
        rows.append(row); print(name, "완료", flush=True)
W = Image.new("RGB", (256 * 8, 256 * 4))
for i, row in enumerate(rows):
    for j, im in enumerate(row):
        W.paste(im, (256 * j, 256 * i))
W.save("eval_results/cmp/sdxl_gen_view.png"); print("저장 eval_results/cmp/sdxl_gen_view.png (행: orig_fp32, orig_fp16, fix_fp16, resid_fp16)")
