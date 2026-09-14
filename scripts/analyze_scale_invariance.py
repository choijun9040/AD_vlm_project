"""
표현 범위와 동적 범위의 직교성 분석 (5장 B)
==============================================
"align도 결국 이상치 억제 기법 아닌가"라는 반론에 답한다.

두 부분으로 구성된다.

(1) **스케일 불변성 실측**
    대칭 INT8 양자화는 scale = amax/127로 절대 크기를 완전히 흡수하므로 SNR이
    스케일에 불변이다. 실제 blk31 활성 텐서에 상수를 곱해가며 이를 확인한다.
    같은 텐서의 fp16 여유는 그 상수에 반비례해 달라진다 — 두 지표가 직교함을
    한 표로 보인다.

(2) **순위 상관**
    해상도 스윕(6모델 × 5해상도 = 30점)에서
      - 절대 크기 ↔ fp16 붕괴율
      - 이상치 비율 ↔ fp16 붕괴율
    의 Spearman 상관을 계산한다. 전자만 붕괴를 설명해야 한다.

실행:
    python scripts/analyze_scale_invariance.py --n_images 20
"""

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images
from profile_vision_activations import (
    DRIVELM_VAL_JSON, BASE_3B, MAX_PIXELS, MIN_PIXELS, build_pixel_cache, get_visual,
)
from profile_int8_quant_error import int8_snr

FP16_MAX = 65504.0
SCALES = [1.0, 1 / 2.7, 1 / 10, 10.0]      # 원래 크기, full과 동일 크기, 1/10, 10배


def spearman(x, y):
    """순위 상관. 동순위는 평균 순위로 처리한다."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


@torch.no_grad()
def part1(n_images):
    print("=" * 78)
    print("(1) 스케일 불변성 — 같은 텐서를 상수배 했을 때 INT8 SNR과 fp16 여유")
    print("=" * 78)
    awq_compat.patch()
    processor = AutoProcessor.from_pretrained(BASE_3B, max_pixels=MAX_PIXELS,
                                              min_pixels=MIN_PIXELS)
    tmap = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in tmap.values() if "CAM_FRONT" in v][:n_images]
    cache = build_pixel_cache(processor, paths, 8)

    out = {}
    for name, adapter in [("student_baseline_v2", "checkpoints/student_baseline_v2/epoch_1"),
                          ("student_full", "checkpoints/student_full/epoch_1")]:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_3B, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter).to("cuda").eval()
        visual = get_visual(model)
        rec = {}
        h = visual.blocks[31].register_forward_hook(
            lambda m, i, o: rec.__setitem__("x", (o[0] if isinstance(o, tuple) else o).detach()))

        rows = []
        for pv, grid in cache:
            visual(pv.to("cuda", torch.bfloat16), grid.to("cuda"))
            x = rec["x"].float()
            rows.append([(s, x.abs().max().item() * s,
                          int8_snr(x * s, per_channel=False),
                          int8_snr(x * s, per_channel=True)) for s in SCALES])
        h.remove(); del model; torch.cuda.empty_cache()

        print(f"\n[{name}]  blk31, 이미지 {len(rows)}장 중앙값")
        print(f"  {'배율':>8}{'max|x|':>12}{'fp16 여유':>11}{'SNR/tensor':>12}{'SNR/chan':>11}")
        print("  " + "-" * 54)
        agg = []
        for i, s in enumerate(SCALES):
            med = lambda f: sorted(r[i][f] for r in rows)[len(rows) // 2]
            mx, st, sc = med(1), med(2), med(3)
            agg.append({"scale": s, "max": mx, "headroom": FP16_MAX / mx,
                        "snr_tensor": st, "snr_channel": sc})
            print(f"  {s:>8.3f}{mx:>12,.0f}{FP16_MAX/mx:>11.2f}{st:>11.2f}dB{sc:>10.2f}dB")
        out[name] = agg
    return out


def part2():
    print("\n" + "=" * 78)
    print("(2) 순위 상관 — 무엇이 fp16 붕괴를 설명하는가")
    print("=" * 78)
    # 해상도 스윕: 절대 크기와 붕괴율
    mag, collapse, labels = [], [], []
    for f in sorted(glob.glob("eval_results/resolution_sweep/res_*.json")):
        mp = int(re.search(r"res_(\d+)", f).group(1))
        d = json.load(open(f))
        for k, v in d.items():
            if k == "_meta":
                continue
            b = v["passes"]["bfloat16"]["by_layer"][str(v["depth"] - 1)]
            mag.append(b["p50"])
            collapse.append(v["passes"]["float16"]["nan_rate"])
            labels.append(f"{k}@{mp//784}")
    rho_mag = spearman(mag, collapse)
    print(f"\n  절대 크기(blk31 p50) ↔ 붕괴율 : Spearman ρ = {rho_mag:+.3f}   (n={len(mag)})")

    # 이상치 비율은 평가 해상도에서만 측정됨 → 6모델 기준 원본 해상도 붕괴율과 대조
    try:
        q = json.load(open("eval_results/int8_quant_error.json"))
        native = json.load(open("eval_results/resolution_sweep/res_1440000.json"))
        ratio, coll2, lab2 = [], [], []
        for k, v in q.items():
            if k == "_meta" or k not in native:
                continue
            ratio.append(v["by_layer"]["31"]["ratio"]["p50"])
            coll2.append(native[k]["passes"]["float16"]["nan_rate"])
            lab2.append(k)
        rho_ratio = spearman(ratio, coll2)
        print(f"  이상치 비율(blk31 max/rms) ↔ 붕괴율 : Spearman ρ = {rho_ratio:+.3f}   (n={len(ratio)})")
        print("\n  모델별 대조 (원본 해상도 붕괴율 기준)")
        print(f"    {'모델':22s}{'blk31 max':>11}{'max/rms':>10}{'붕괴율':>9}")
        for k, r, c in sorted(zip(lab2, ratio, coll2), key=lambda t: -t[2]):
            m = native[k]["passes"]["bfloat16"]["by_layer"]["31"]["p50"]
            print(f"    {k:22s}{m:>11,.0f}{r:>10.1f}{c*100:>8.1f}%")
    except FileNotFoundError:
        rho_ratio = None
        print("  (INT8 결과 파일 없음 — 이상치 비율 상관 생략)")
    return {"rho_magnitude": rho_mag, "rho_ratio": rho_ratio, "n_points": len(mag)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=20)
    ap.add_argument("--out", default="eval_results/scale_invariance_analysis.json")
    args = ap.parse_args()
    r1 = part1(args.n_images)
    r2 = part2()
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps({"scale_sweep": r1, "correlation": r2},
                                         indent=2, ensure_ascii=False))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
