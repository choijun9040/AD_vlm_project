"""개정 31 — SDXL-VAE-FP16-Fix 원문(토론 #7) 조건으로 원래 SDXL-VAE·FP16-Fix·(R)을 잰다. 2026-11-06, 결과 보기 전 등록(11:47 KST).
COCO val2017 5,000장, RandomCrop(256, pad_if_needed=True)·시드 = 번호, 인코더 입력 [-1,1], 분포 평균으로 인코딩, 각 형식으로 인코딩·디코딩.
LPIPS(alex, [-1,1]), SSIM(skimage, [0,1], channel_axis=-1, data_range=1), MSE([0,1]) — 원본 사진 대비. 평균 [10.5·89.5 백분위]. 원래 VAE fp32 출력 대비 PSNR도 함께.
(R) = 배포하는 같은 교정 디코더(c = 1/64, eps ×c²) — 인코더는 원래 인코더를 그 형식으로. 문항마다 덧붙여 끊겨도 이어 간다. venv_sd.

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_coco.py --coco /dev/shm/coco/val2017 --out eval_results/cmp/sdxl_coco.json
"""
import argparse, copy, json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_sdxl_vae_r import apply_resid_vae

C = 1 / 64


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--enc_plan", default=None, help="개정 32: 인코더 (R) 계획(json) — 「(R) 인코더 + (R) 디코더, 모두 fp16」 행을 더한다")
    ap.add_argument("--no_tf32", action="store_true", help="개정 39: 행렬곱·합성곱 TF32 끔(기준을 순수 fp32로)")
    ap.add_argument("--dec_c", type=float, default=C, help="개정 33: 디코더 (R)의 c(기본 = 생성 잠재로 정한 1/64)")
    ap.add_argument("--lpips_input", choices=["m11", "01", "both"], default="m11", help="개정 34: LPIPS 입력 범위 — m11 = [-1,1](라이브러리 규약), 01 = [0,1] 그대로(원문 값 재현 설정)")
    ap.add_argument("--rows", default=None, help="잴 행만(쉼표) — orig_fp32는 PSNR 기준이라 늘 포함")
    a = ap.parse_args()
    if a.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    import lpips
    from skimage.metrics import structural_similarity
    from torchvision import transforms
    from diffusers import AutoencoderKL
    files = sorted(Path(a.coco).glob("*.jpg"))[:a.limit]
    crop = transforms.RandomCrop(256, pad_if_needed=True)
    to_t = transforms.ToTensor()
    imgs = []
    for i, f in enumerate(files):
        torch.manual_seed(i)
        imgs.append(to_t(crop(Image.open(f).convert("RGB"))))          # [0,1], 3×256×256
    print(f"[입력] {len(imgs)}장", flush=True)
    orig32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).eval()
    fix32 = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32).eval()
    r32 = copy.deepcopy(orig32)
    with torch.no_grad():
        apply_resid_vae(r32, a.dec_c, eps=True)
    # (이름, VAE, 인코딩 형식, 디코딩 형식)
    cfgs = [("orig_fp32", orig32, torch.float32, torch.float32), ("orig_bf16", orig32, torch.bfloat16, torch.bfloat16),
            ("orig_fp16", orig32, torch.float16, torch.float16), ("fix_fp32", fix32, torch.float32, torch.float32),
            ("fix_bf16", fix32, torch.bfloat16, torch.bfloat16), ("fix_fp16", fix32, torch.float16, torch.float16),
            ("resid_fp16", r32, torch.float16, torch.float16), ("resid_dec_fp16_enc_fp32", r32, torch.float32, torch.float16)]
    if a.enc_plan:
        from sdxl_encoder_resid import apply_plan
        ep = json.loads(Path(a.enc_plan).read_text())
        rf32 = apply_plan(copy.deepcopy(r32), ep["saved"])
        cfgs.append(("resid_full_fp16", rf32, torch.float16, torch.float16))
    if a.rows:
        keep = set(a.rows.split(",")) | {"orig_fp32"}
        cfgs = [c for c in cfgs if c[0] in keep]
    lp = lpips.LPIPS(net="alex").cuda().eval()
    out = Path(a.out); res = {"n": len(imgs), "c": a.dec_c, "protocol": "FP16-Fix 토론 #7 + 등록한 기본값(개정 31)", "lpips_input": a.lpips_input}
    ref_out = {}
    q = lambda v, f: float(np.percentile(v, f)) if v else None
    for name, vae, edt, ddt in cfgs:
        part = Path(str(out) + f".{name}.partial.jsonl"); rows = []
        if part.exists():
            rows = [json.loads(l) for l in part.read_text().splitlines() if l.strip()]
        enc = copy.deepcopy(vae).to("cuda", edt).eval(); dec = enc if edt == ddt else copy.deepcopy(vae).to("cuda", ddt).eval()
        fo = open(part, "a")
        with torch.no_grad():
            for i, x in enumerate(imgs):
                if i < len(rows) and name != "orig_fp32":
                    continue
                xx = (x * 2 - 1)[None].cuda()
                z = enc.encode(xx.to(edt)).latent_dist.mean
                y = dec.decode(z.to(ddt)).sample.float()
                if name == "orig_fp32":
                    ref_out[i] = y.cpu()
                if i < len(rows):
                    continue
                fin = bool(torch.isfinite(y).all()) and bool(torch.isfinite(z).all())
                r = {"i": i, "finite": fin, "enc_finite": bool(torch.isfinite(z).all())}
                if fin:
                    y01 = ((y.clamp(-1, 1) + 1) / 2)[0]; x01 = x.cuda()
                    if a.lpips_input == "both":                                   # 개정 38: 두 설정을 한 번에
                        r["lpips"] = float(lp(y.clamp(-1, 1), xx)); r["lpips01"] = float(lp(y01[None], x01[None]))
                    else:
                        r["lpips"] = float(lp(y.clamp(-1, 1), xx)) if a.lpips_input == "m11" else float(lp(y01[None], x01[None]))
                    r["mse"] = float((y01 - x01).pow(2).mean())
                    r["ssim"] = float(structural_similarity(x01.permute(1, 2, 0).cpu().numpy(), y01.permute(1, 2, 0).cpu().numpy(), channel_axis=-1, data_range=1.0))
                    if i in ref_out and name != "orig_fp32":
                        m = float((((y.clamp(-1, 1) + 1) / 2) - ((ref_out[i].cuda().clamp(-1, 1) + 1) / 2)).pow(2).mean())
                        r["psnr_vs_orig32"] = 10 * math.log10(1 / m) if m > 0 else float("inf")
                fo.write(json.dumps(r) + "\n"); fo.flush(); rows.append(r)
        del enc, dec; torch.cuda.empty_cache()
        ok = [r for r in rows if r["finite"]]
        summ = {"nonfinite": len(rows) - len(ok), "enc_nonfinite": sum(1 for r in rows if not r["enc_finite"])}
        for k in ("lpips", "lpips01", "ssim", "mse", "psnr_vs_orig32"):
            v = [r[k] for r in ok if k in r and math.isfinite(r[k])]
            if v:
                summ[k] = {"mean": float(np.mean(v)), "ci79": [q(v, 10.5), q(v, 89.5)]}
        res[name] = summ
        print(f"[{name}] {json.dumps(summ, ensure_ascii=False)}", flush=True)
        Path(str(out) + ".partial").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))       # 다 끝난 뒤에만
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
