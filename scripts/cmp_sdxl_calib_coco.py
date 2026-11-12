"""개정 33 — SDXL 디코더 (R)의 c를 COCO train2017 100장(평가 val2017과 겹치지 않음)으로 정한다. 2026-11-06, 결과 보기 전 등록(12:16 KST).
측정용 사진 → RandomCrop 256(시드 = 번호, cmp_sdxl_coco.py와 같다) → 원래 VAE fp32 인코더 분포 평균 → post_quant_conv+디코더 bf16 연산 단위 측정 → 최댓값 → c(개정 20과 같은 방식).
목표 초과 연산이 배포 디코더 영역(up_blocks.2 업샘플러 ~ up_blocks.3) 밖에 있으면 따로 적는다. venv_sd.

    /workspace/venv_sd/bin/python scripts/cmp_sdxl_calib_coco.py --images /dev/shm/coco/train100 --out eval_results/cmp/sdxl_calib_coco.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                          # noqa: E402

FP16_MAX, TARGET = 65504.0, 2.0
REGION = ("up_blocks.2.upsamplers", "up_blocks.3", "conv_norm_out")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from diffusers import AutoencoderKL
    from torchvision import transforms
    from PIL import Image
    files = sorted(Path(a.images).glob("*.jpg"))
    crop = transforms.RandomCrop(256, pad_if_needed=True)
    to_t = transforms.ToTensor()
    vae32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    lat = []
    with torch.no_grad():
        for i, f in enumerate(files):
            torch.manual_seed(i)
            x = (to_t(crop(Image.open(f).convert("RGB"))) * 2 - 1)[None].cuda()
            lat.append(vae32.encode(x).latent_dist.mean.cpu())
    print(f"[측정용] {len(lat)}장 → 잠재 {tuple(lat[0].shape)}", flush=True)
    mb = vae32.to(torch.bfloat16)
    with torch.no_grad():
        cache = [mb.post_quant_conv(z.cuda().to(torch.bfloat16)) for z in lat]
        ops = HX.profile_ops(mb.decoder, cache, lambda m, x: m(x), FP16_MAX, TARGET)
    g = ops["global_max"]
    c = HX.choose_c(g["max_worst"], FP16_MAX, TARGET)
    over = [(r["point"], r["max_worst"]) for r in ops["rows"] if FP16_MAX / TARGET < r["max_worst"] < 1e30]
    outside = [p for p in over if not any(k in p[0] for k in REGION)]
    res = {"images": f"COCO train2017 {files[0].name}~{files[-1].name} ({len(files)}장)", "max_worst": g["max_worst"], "point": g["point"],
           "any_exceed_rate": ops.get("any_exceed_rate"), "first_exceed": (ops["first_exceed"] or {}).get("point"), "c": c,
           "n_over_target": len(over), "over_outside_region": outside}
    print(f"[측정용] 최대 {g['max_worst']:,.0f} @ {g['point']} · 초과 비율 {ops.get('any_exceed_rate')} · c = {c} · 목표 초과 {len(over)} · 영역 밖 {outside[:5]}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
