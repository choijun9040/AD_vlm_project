"""
C* 2단계 — SDXL VAE에 측정 → (R) (개정 16, 구성은 2026-11-04 18:28 KST 결과 전 등록).
입구 up_blocks.2.upsamplers.0.conv ×c, 쓰는 층 up_blocks.3.resnets[k].conv2 ×c, 지름길 up_blocks.3.resnets[0].conv_shortcut 편향 ×c,
흐름을 읽는 GroupNorm(up_blocks.3.resnets[k].norm1, conv_norm_out) eps ×c². c는 1단계 측정 최댓값에서(목표 여유 2, 2의 거듭제곱).
fp32 등가성(799장) → fp16 비유한·원래 fp32 대비 오차·PSNR → bf16 재진단(확인).

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_vae_r.py --stage1 eval_results/cmp/sdxl_vae_stage1.json --out eval_results/cmp/sdxl_vae_resid.json
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                          # noqa: E402
from dataloader import build_token_to_images      # noqa: E402
from cmp_sdxl_vae import load_img, VAL, FP16_MAX, TARGET   # noqa: E402


@torch.no_grad()
def apply_resid_vae(vae, c, eps=True):
    d = vae.decoder
    up = d.up_blocks[2].upsamplers[0].conv
    up.weight.mul_(c); up.bias.mul_(c)
    for k, r in enumerate(d.up_blocks[3].resnets):
        r.conv2.weight.mul_(c); r.conv2.bias.mul_(c)
        if r.conv_shortcut is not None:
            r.conv_shortcut.bias.mul_(c)
        if eps:
            r.norm1.eps = r.norm1.eps * c * c
    if eps:
        d.conv_norm_out.eps = d.conv_norm_out.eps * c * c


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_eps", action="store_true", help="개정 18 ② — GroupNorm eps를 바꾸지 않는다(가중치 파일만으로 배포)")
    ap.add_argument("--skip_rediag", action="store_true")
    a = ap.parse_args()
    from diffusers import AutoencoderKL
    s1 = json.loads(Path(a.stage1).read_text())
    w, h = s1["size"]
    mx = s1["diagnosis_brief"]["max_worst"]
    c = HX.choose_c(mx, FP16_MAX, TARGET)
    orig = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    tok = build_token_to_images(VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]
    lat, ref = [], []
    with torch.no_grad():
        for p in paths:
            z = orig.encode(load_img(p, w, h).cuda()).latent_dist.mean
            lat.append(z.cpu()); ref.append(orig.decode(z).sample.float().cpu())
    r32 = copy.deepcopy(orig)
    apply_resid_vae(r32, c, eps=not a.no_eps)
    cos = []
    with torch.no_grad():
        for z, y in zip(lat, ref):
            o = r32.decode(z.cuda()).sample.float().cpu()
            cos.append(float(torch.nn.functional.cosine_similarity(o.flatten().double(), y.flatten().double(), dim=0)))
    res = {"c": c, "eps_adjusted": not a.no_eps, "measured_max": mx, "n": len(paths), "equivalence_fp32": {"cos_min": min(cos), "cos_median": sorted(cos)[len(cos) // 2]},
           "accepted": min(cos) >= 0.99999}
    print(f"[(R)] c={c} · fp32 등가성 코사인 최소 {min(cos):.8f}", flush=True)
    r16 = copy.deepcopy(r32).half()
    bad, errs, psnr = 0, [], []
    with torch.no_grad():
        for z, y in zip(lat, ref):
            o = r16.decode(z.cuda().half()).sample.float().cpu()
            if not torch.isfinite(o).all():
                bad += 1; continue
            errs.append(float((o.double() - y.double()).norm() / y.double().norm()))
            mse = float((((o + 1) / 2).clamp(0, 1) - ((y + 1) / 2).clamp(0, 1)).pow(2).mean())
            psnr.append(10 * math.log10(1 / mse) if mse > 0 else float("inf"))
    fe, fp = sorted(errs), sorted(psnr)
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
    res["resid_fp16"] = {"nonfinite": bad, "rate": bad / len(lat), "rel_err": {"median": q(fe, .5), "p99": q(fe, .99)},
                         "psnr_db": {"median": q(fp, .5), "min": fp[0] if fp else None}}
    print(f"[(R) fp16] 비유한 {bad}/{len(lat)} · 상대 오차 중앙 {q(fe, .5)} · PSNR 중앙 {q(fp, .5)}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    if a.skip_rediag:
        print(f"저장: {a.out}")
        return
    # 재진단(확인) — bf16
    rb = copy.deepcopy(r32).to(torch.bfloat16)
    cache = [rb.post_quant_conv(z.cuda().to(torch.bfloat16)) for z in lat]
    ops = HX.profile_ops(rb.decoder, cache, lambda m, x: m(x), FP16_MAX, TARGET)
    g = ops["global_max"]
    res["rediagnosis_bf16"] = {"global_max_point": g["point"], "max_worst": g["max_worst"], "headroom_worst": g["headroom_worst"],
                               "first_exceed": ops["first_exceed"], "any_exceed_rate": ops.get("any_exceed_rate")}
    print(f"[재진단] 최대 {g['max_worst']:,.0f} @ {g['point']} · 최악 여유 {g['headroom_worst']:.2f}배 · 초과 비율 {ops.get('any_exceed_rate')}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
