"""
C* — SDXL-VAE-FP16-Fix 원문 결과물과의 비교 (1단계: 진단·fp16 상실·FP16-Fix). 2026-11-04.
사전 등록: thesis_outline_20260910.md §6 개정 16 (결과 보기 전).

원래 VAE(stabilityai/sdxl-vae)와 원문이 공개한 미세조정 VAE(madebyollin/sdxl-vae-fp16-fix)를 그대로 쓴다.
입력: DriveLM 검증 CAM_FRONT 799장(표준 순서)을 1024×576으로 줄여 원래 VAE 인코더(fp32, 분포 평균)로 만든 잠재.
  ① 원래 디코더 fp16 출력의 비유한 비율
  ② 측정 — 원래 디코더를 bf16에서 연산 단위로 재(headroom_ext.profile_ops) 최악 여유·초과 비율·처음 넘는 연산을 얻고 ①과 대조
  ④ FP16-Fix 디코더 fp16의 비유한 비율과 원래 디코더 fp32 출력 대비 차이(상대 오차, PSNR — 출력은 [-1,1]을 [0,1]로)
venv_sd(diffusers만 따로, venv_main 패키지를 읽기만)에서 돈다.

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_vae.py --out eval_results/cmp/sdxl_vae_stage1.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                          # noqa: E402
from dataloader import build_token_to_images      # noqa: E402

VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
FP16_MAX = 65504.0
TARGET = 2.0


def load_img(p, w, h):
    im = Image.open(p).convert("RGB").resize((w, h), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]
    return x


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from diffusers import AutoencoderKL

    orig = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    tok = build_token_to_images(VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]

    # 잠재(원래 인코더 fp32)와 원래 디코더 fp32 출력(기준)
    lat, ref = [], []
    with torch.no_grad():
        for p in paths:
            z = orig.encode(load_img(p, a.width, a.height).cuda()).latent_dist.mean
            lat.append(z.cpu())
            ref.append(orig.decode(z).sample.float().cpu())
    res = {"n": len(paths), "size": [a.width, a.height], "latent_shape": list(lat[0].shape)}
    print(f"[준비] 잠재 {len(lat)}개 {tuple(lat[0].shape)}", flush=True)

    def fp16_eval(vae16, name):
        bad, errs, psnr = 0, [], []
        with torch.no_grad():
            for z, y in zip(lat, ref):
                o = vae16.decode(z.cuda().half()).sample.float().cpu()
                if not torch.isfinite(o).all():
                    bad += 1; continue
                errs.append(float((o.double() - y.double()).norm() / y.double().norm()))
                mse = float((((o + 1) / 2).clamp(0, 1) - ((y + 1) / 2).clamp(0, 1)).pow(2).mean())
                psnr.append(10 * math.log10(1.0 / mse) if mse > 0 else float("inf"))
        fe, fp = sorted(errs), sorted(psnr)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
        out = {"nonfinite": bad, "rate": bad / len(lat), "rel_err": {"median": q(fe, .5), "p99": q(fe, .99)},
               "psnr_db": {"median": q(fp, .5), "min": fp[0] if fp else None}}
        print(f"[{name}] fp16 비유한 {bad}/{len(lat)} · 상대 오차 중앙 {out['rel_err']['median']} · PSNR 중앙 {out['psnr_db']['median']}", flush=True)
        return out

    # ① 원래 디코더 fp16
    o16 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda").eval()
    res["orig_fp16"] = fp16_eval(o16, "원래 VAE")
    del o16; torch.cuda.empty_cache()
    # ④ FP16-Fix 디코더 fp16 (원문 결과물 그대로)
    f16 = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda").eval()
    res["fix_fp16"] = fp16_eval(f16, "FP16-Fix")
    del f16; torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

    # ② 측정 — bf16 연산 단위 크기(디코더만)
    ob = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.bfloat16).to("cuda").eval()
    dec = ob.decoder
    cache = [ob.post_quant_conv(z.cuda().to(torch.bfloat16)) for z in lat]       # decode()와 같게 post_quant_conv 뒤를 디코더 입력으로
    ops = HX.profile_ops(dec, cache, lambda m, x: m(x), FP16_MAX, TARGET)
    res["diagnosis_bf16"] = HX.summarize_ops(ops)
    g = ops["global_max"]
    res["diagnosis_brief"] = {"global_max_point": g["point"], "max_worst": g["max_worst"], "headroom_worst": g["headroom_worst"],
                              "exceed_rate_at_global": g.get("exceed_rate"), "first_exceed": ops.get("first_exceed"),
                              "any_exceed_rate_predicted_loss": ops.get("any_exceed_rate")}
    print(f"[측정] 최대 {g['max_worst']:,.0f} @ {g['point']} · 최악 여유 {g['headroom_worst']:.3f}배 · 이미지별 초과 비율(상실 추정) {ops.get('any_exceed_rate')}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
