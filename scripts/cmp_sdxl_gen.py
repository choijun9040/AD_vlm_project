"""
C* 개정 18 ① — SDXL이 생성한 잠재에서 원래 VAE·FP16-Fix·(R) 비교 (FP16-Fix의 실제 쓰임). 2026-11-04, 결과 보기 전 등록(19:05 KST).
SDXL base 1.0(fp16 UNet·텍스트 인코더), PartiPrompts 앞 100개, 1024×1024, 30스텝, 안내 5.0, 시드 = 번호. (R)의 c는 DriveLM에서 잰 1/64를 그대로 쓴다.
비교: 원래 VAE fp32(기준)·fp16, FP16-Fix fp16, (R) fp16(eps 조정·미조정). 함께: 생성 잠재에서의 측정(bf16 최댓값·이미지별 초과 비율)과 (R) 재진단.

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_gen.py --tsv data/PartiPrompts.tsv --out eval_results/cmp/sdxl_gen_cmp.json
"""
import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                          # noqa: E402
from cmp_sdxl_vae_r import apply_resid_vae        # noqa: E402

FP16_MAX, TARGET, C_FIXED = 65504.0, 2.0, 1 / 64


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_tf32", action="store_true", help="개정 39: 행렬곱·합성곱 TF32 끔(기준을 순수 fp32로)")
    ap.add_argument("--c", type=float, default=None, help="개정 38: 잔차 재조정 c(기본 = 1/64 고정값). 새 환경의 측정용 c를 넣는다")
    a = ap.parse_args()
    if a.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    with open(a.tsv, newline="") as f:
        prompts = [r["Prompt"] for r in csv.DictReader(f, delimiter="\t")][:a.n]
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
                       generator=torch.Generator("cuda").manual_seed(i), output_type="latent")
            lat.append((out.images.float() / vae32.config.scaling_factor).cpu())     # 디코더 입력(스케일 되돌림)
            if (i + 1) % 20 == 0:
                print(f"  생성 {i + 1}/{len(prompts)}", flush=True)
        torch.save(lat, lat_path)
        del pipe; torch.cuda.empty_cache()
    global C_FIXED
    if a.c is not None:
        C_FIXED = a.c
    res = {"n": len(lat), "c": C_FIXED, "prompts": "PartiPrompts 앞 100개", "gen": {"steps": 30, "cfg": 5.0, "size": 1024}}
    with torch.no_grad():
        ref = [vae32.decode(z.cuda()).sample.float().cpu() for z in lat]

    def ev(vae16, name):
        bad, errs, psnr = 0, [], []
        with torch.no_grad():
            for z, y in zip(lat, ref):
                o = vae16.decode(z.cuda().half()).sample.float().cpu()
                if not torch.isfinite(o).all():
                    bad += 1; continue
                errs.append(float((o.double() - y.double()).norm() / y.double().norm()))
                mse = float((((o + 1) / 2).clamp(0, 1) - ((y + 1) / 2).clamp(0, 1)).pow(2).mean())
                psnr.append(10 * math.log10(1 / mse) if mse > 0 else float("inf"))
        fe, fp = sorted(errs), sorted(psnr)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
        res[name] = {"nonfinite": bad, "rate": bad / len(lat), "rel_err": {"median": q(fe, .5), "p99": q(fe, .99)},
                     "psnr_db": {"median": q(fp, .5), "min": fp[0] if fp else None}}
        print(f"[{name}] 비유한 {bad}/{len(lat)} · 상대 오차 중앙 {q(fe, .5)} · PSNR 중앙 {q(fp, .5)}", flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

    ev(copy.deepcopy(vae32).half(), "orig_fp16")
    ev(AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda").eval(), "fix_fp16")
    for eps in (True, False):
        r = copy.deepcopy(vae32); apply_resid_vae(r, C_FIXED, eps=eps)
        with torch.no_grad():
            cos = [float(torch.nn.functional.cosine_similarity(r.decode(z.cuda()).sample.double().flatten(), y.double().cuda().flatten(), dim=0))
                   for z, y in zip(lat, ref)]
        res[f"resid_eps{int(eps)}_equivalence_cos_min"] = min(cos)
        ev(copy.deepcopy(r).half(), f"resid_eps{int(eps)}_fp16")
        del r; torch.cuda.empty_cache()
    # 측정(원래 디코더, bf16)과 (R) 재진단 — 생성 잠재
    for name, model in (("orig", copy.deepcopy(vae32)), ("resid", None)):
        if model is None:
            model = copy.deepcopy(vae32); apply_resid_vae(model, C_FIXED)
        mb = model.to(torch.bfloat16)
        with torch.no_grad():
            cache = [mb.post_quant_conv(z.cuda().to(torch.bfloat16)) for z in lat]
        ops = HX.profile_ops(mb.decoder, cache, lambda m, x: m(x), FP16_MAX, TARGET)
        g = ops["global_max"]
        res[f"diag_{name}"] = {"max_worst": g["max_worst"], "point": g["point"], "headroom_worst": g["headroom_worst"],
                               "any_exceed_rate": ops.get("any_exceed_rate"), "first_exceed": (ops["first_exceed"] or {}).get("point")}
        print(f"[측정 {name}] 최대 {g['max_worst']:,.0f} @ {g['point']} · 최악 여유 {g['headroom_worst']:.3f} · 초과 비율 {ops.get('any_exceed_rate')}", flush=True)
        del mb, model, cache; torch.cuda.empty_cache()
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
