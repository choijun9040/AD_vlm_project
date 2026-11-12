"""
C* (G)·(A) 판단용 — 원래 SDXL VAE 디코더를 bf16에서 연산 단위로 재 목표 여유(32,752) 초과 연산 전부를 종류로 나눈다 (개정 16·17, 2026-11-04 18:37 KST 등록 뒤).
종류: stream(마지막 업 블록 잔차·업샘플 합성곱·지름길) / resnet_inner(conv1·norm2·silu) / attention(mid_block.attentions) / other.

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_classify.py --out eval_results/cmp/sdxl_vae_classify.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                          # noqa: E402
from dataloader import build_token_to_images      # noqa: E402
from cmp_sdxl_vae import load_img, VAL, FP16_MAX, TARGET   # noqa: E402


def kind(point):
    mod = point.split("::")[0]
    if ".attentions." in mod or mod.startswith("mid_block.attentions"):
        return "attention"
    if any(k in mod for k in (".conv1", ".norm2", ".nonlinearity")) and ".resnets." in mod:
        return "resnet_inner"
    if ".resnets." in mod or ".upsamplers." in mod or mod in ("conv_in", "conv_norm_out", "conv_out"):
        return "stream"
    return "other"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from diffusers import AutoencoderKL
    v32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    tok = build_token_to_images(VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]
    with torch.no_grad():
        lat = [v32.encode(load_img(p, 1024, 576).cuda()).latent_dist.mean for p in paths]
    vb = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.bfloat16).to("cuda").eval()
    with torch.no_grad():
        cache = [vb.post_quant_conv(z.to(torch.bfloat16)) for z in lat]
    ops = HX.profile_ops(vb.decoder, cache, lambda m, x: m(x), FP16_MAX, TARGET)
    over = [r for r in ops["rows"] if r["max_worst"] > FP16_MAX / TARGET]
    by = {}
    for r in over:
        by.setdefault(kind(r["point"]), []).append({"point": r["point"], "max_worst": r["max_worst"], "exceed_rate": r["exceed_rate"]})
    res = {"n": len(paths), "target": FP16_MAX / TARGET, "n_over_target": len(over),
           "by_kind": {k: sorted(v, key=lambda t: -t["max_worst"]) for k, v in by.items()},
           "counts": {k: len(v) for k, v in by.items()}}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res["counts"], ensure_ascii=False))
    for k, v in res["by_kind"].items():
        print(k, [(t["point"], round(t["max_worst"])) for t in v[:6]])


if __name__ == "__main__":
    main()
