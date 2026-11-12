"""개정 41 — SDXL VAE 디코더 bf16, 생성 잠재 100개 (그림 5-3·표 [생성-디코더]의 bf16 행) + 같은 환경 확인. 2026-11-11, 결과 보기 전 등록.
개정 39(env3)와 같은 잠재(sdxl_gen_cmp.latents.pt)·같은 기준(원래 VAE fp32, TF32 끔)·같은 지표(복호 이미지 상대 오차, [0,1] PSNR)·같은 모델 판.
구성: 원래 VAE bf16, FP16-Fix bf16 + 확인용 FP16-Fix fp16, 잔차 재조정 fp16(c = 1/64, eps ×c²). 상대 오차는 중앙·p99·최대를 기록한다.
venv_sd(torch 2.11.0+cu128, diffusers 0.41.0).

    python scripts/sdxl_gen_bf16.py --latents eval_results/cmp/env5/sdxl_gen_cmp.latents.pt --out eval_results/cmp/env5/sdxl_gen_bf16.json
"""
import argparse, copy, json, math
from pathlib import Path
import torch

REV_VAE, REV_FIX, C = "6f5909a7", "207b116d", 1 / 64


@torch.no_grad()
def apply_resid_vae(vae, c, eps=True):                  # scripts/cmp_sdxl_vae_r.py와 같다(의존 모듈 없이 옮김)
    d = vae.decoder
    up = d.up_blocks[2].upsamplers[0].conv
    up.weight.mul_(c); up.bias.mul_(c)
    for r in d.up_blocks[3].resnets:
        r.conv2.weight.mul_(c); r.conv2.bias.mul_(c)
        if r.conv_shortcut is not None:
            r.conv_shortcut.bias.mul_(c)
        if eps:
            r.norm1.eps = r.norm1.eps * c * c
    if eps:
        d.conv_norm_out.eps = d.conv_norm_out.eps * c * c


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import diffusers
    from diffusers import AutoencoderKL
    lat = torch.load(a.latents)
    vae32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", revision=REV_VAE, torch_dtype=torch.float32).to("cuda").eval()
    with torch.no_grad():
        ref = [vae32.decode(z.cuda()).sample.float().cpu() for z in lat]
    res = {"n": len(lat), "latents": a.latents, "tf32": False, "torch": torch.__version__, "diffusers": diffusers.__version__,
           "gpu": torch.cuda.get_device_name(0), "rev_vae": REV_VAE, "rev_fix": REV_FIX, "c": C}

    def ev(vae, dt, name):
        bad, errs, psnr = 0, [], []
        with torch.no_grad():
            for z, y in zip(lat, ref):
                o = vae.decode(z.cuda().to(dt)).sample.float().cpu()
                if not torch.isfinite(o).all():
                    bad += 1; continue
                errs.append(float((o.double() - y.double()).norm() / y.double().norm()))
                mse = float((((o + 1) / 2).clamp(0, 1) - ((y + 1) / 2).clamp(0, 1)).pow(2).mean())
                psnr.append(10 * math.log10(1 / mse) if mse > 0 else float("inf"))
        fe, fp = sorted(errs), sorted(psnr)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
        res[name] = {"nonfinite": bad, "rate": bad / len(lat), "rel_err": {"median": q(fe, .5), "p99": q(fe, .99), "max": fe[-1] if fe else None},
                     "psnr_db": {"median": q(fp, .5), "min": fp[0] if fp else None}}
        print(f"[{name}] 비유한 {bad}/{len(lat)} · 상대 오차 중앙 {q(fe, .5)} · 최대 {fe[-1] if fe else None} · PSNR 중앙 {q(fp, .5)}", flush=True)
        Path(a.out + ".partial").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    ev(copy.deepcopy(vae32).to(torch.bfloat16), torch.bfloat16, "orig_bf16")
    fix = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", revision=REV_FIX, torch_dtype=torch.float32).to("cuda").eval()
    ev(copy.deepcopy(fix).to(torch.bfloat16), torch.bfloat16, "fix_bf16")
    ev(copy.deepcopy(fix).half(), torch.float16, "fix_fp16")                          # 확인용 — 개정 39 값과 비교
    r = copy.deepcopy(vae32); apply_resid_vae(r, C, eps=True)
    with torch.no_grad():
        res["resid_equivalence_cos_min"] = min(float(torch.nn.functional.cosine_similarity(r.decode(z.cuda()).sample.double().flatten(),
                                                                                           y.double().cuda().flatten(), dim=0)) for z, y in zip(lat, ref))
    ev(copy.deepcopy(r).half(), torch.float16, "resid_fp16")                            # 확인용 — 개정 39 값과 비교
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
