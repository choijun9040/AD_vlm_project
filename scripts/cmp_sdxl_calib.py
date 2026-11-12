"""
개정 20 — SDXL VAE (R)의 c를 SDXL 생성 잠재(측정용, 평가와 겹치지 않음)로 정한다. 2026-11-04, 결과 보기 전 등록.
측정용: PartiPrompts 101~200번(시드 = 번호 100~199), 생성 설정은 평가용(cmp_sdxl_gen.py, 1~100번)과 같다. 원래 디코더 bf16 연산 단위 측정 → 최댓값 → c.

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_calib.py --tsv data/PartiPrompts.tsv --out eval_results/cmp/sdxl_calib.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                          # noqa: E402

FP16_MAX, TARGET = 65504.0, 2.0


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    with open(a.tsv, newline="") as f:
        prompts = [r["Prompt"] for r in csv.DictReader(f, delimiter="\t")][a.start:a.start + a.n]
    lat_path = Path(a.out).with_suffix(".latents.pt")
    vae32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    if lat_path.exists():
        lat = torch.load(lat_path)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
                                                         variant="fp16", use_safetensors=True, vae=vae32).to("cuda")
        lat = []
        for i, p in enumerate(prompts):
            out = pipe(prompt=p, num_inference_steps=30, guidance_scale=5.0, height=1024, width=1024,
                       generator=torch.Generator("cuda").manual_seed(a.start + i), output_type="latent")
            lat.append((out.images.float() / vae32.config.scaling_factor).cpu())
            if (i + 1) % 20 == 0:
                print(f"  생성 {i + 1}/{len(prompts)}", flush=True)
        torch.save(lat, lat_path)
        del pipe; torch.cuda.empty_cache()
    mb = vae32.to(torch.bfloat16)
    with torch.no_grad():
        cache = [mb.post_quant_conv(z.cuda().to(torch.bfloat16)) for z in lat]
    ops = HX.profile_ops(mb.decoder, cache, lambda m, x: m(x), FP16_MAX, TARGET)
    g = ops["global_max"]
    c = HX.choose_c(g["max_worst"], FP16_MAX, TARGET)
    res = {"prompts": f"PartiPrompts {a.start + 1}~{a.start + len(prompts)}번", "seeds": [a.start, a.start + len(prompts) - 1],
           "max_worst": g["max_worst"], "point": g["point"], "any_exceed_rate": ops.get("any_exceed_rate"),
           "first_exceed": (ops["first_exceed"] or {}).get("point"), "c": c}
    print(f"[측정용] 최대 {g['max_worst']:,.0f} @ {g['point']} · 초과 비율 {ops.get('any_exceed_rate')} · c = {c}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
