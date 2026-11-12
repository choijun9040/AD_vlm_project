"""개정 34 — LPIPS 원문 값(원래 VAE 0.055·FP16-Fix 0.056) 재현 시험. 2026-11-06, 결과 보기 전 등록(14:23 KST).
COCO val2017 앞 N장(cmp_sdxl_coco.py와 같은 자르기·시드)을 원래 VAE·FP16-Fix fp32로 복원하고 LPIPS를 alex·vgg × 입력 [-1,1]·[0,1]로 잰다. venv_sd.

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_lpips_probe.py --coco /dev/shm/coco/val2017 --n 1000 --out eval_results/cmp/sdxl_lpips_probe.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import lpips
    from torchvision import transforms
    from diffusers import AutoencoderKL
    files = sorted(Path(a.coco).glob("*.jpg"))[:a.n]
    crop = transforms.RandomCrop(256, pad_if_needed=True); to_t = transforms.ToTensor()
    nets = {k: lpips.LPIPS(net=k).cuda().eval() for k in ("alex", "vgg")}
    vaes = {"orig": "stabilityai/sdxl-vae", "fix": "madebyollin/sdxl-vae-fp16-fix"}
    res = {"n": len(files), "original_paper": {"orig": 0.055, "fix": 0.056}}
    for vn, repo in vaes.items():
        vae = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float32).cuda().eval()
        vals = {f"{k}_{r}": [] for k in nets for r in ("m11", "01")}
        with torch.no_grad():
            for i, f in enumerate(files):
                torch.manual_seed(i)
                x = to_t(crop(Image.open(f).convert("RGB")))[None].cuda()          # [0,1]
                xx = x * 2 - 1
                y = vae.decode(vae.encode(xx).latent_dist.mean).sample.clamp(-1, 1)
                y01 = (y + 1) / 2
                for k, net in nets.items():
                    vals[f"{k}_m11"].append(float(net(y, xx)))                   # 라이브러리 규약 [-1,1]
                    vals[f"{k}_01"].append(float(net(y01, x)))                   # [0,1]을 그대로 넣은 경우
        res[vn] = {k: float(np.mean(v)) for k, v in vals.items()}
        print(f"[{vn}] " + " · ".join(f"{k} {v:.4f}" for k, v in res[vn].items()), flush=True)
        del vae; torch.cuda.empty_cache()
    hit = [k for k in res["orig"] if abs(res["orig"][k] - 0.055) <= 0.003 and abs(res["fix"][k] - 0.056) <= 0.003]
    res["reproduces"] = hit
    print(f"[판정] 재현 설정: {hit}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
