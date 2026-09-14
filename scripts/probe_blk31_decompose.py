"""
blk31 massive activation 보강 실험 — 5변형 확대 · 성분 분해 · 정규화 흡수 확인
================================================================================
`probe_blk31_intervention.py`가 2모델·30장에서 "blk31은 마지막 블록의 자체 생성"임을
보였다. 세 가지를 보강한다.

1. **5변형 전체로 확대** — baseline_v2 / kd_only_v4 / temporal / spatial / full
2. **성분 분해** — 비전 블록은 `h = h + attn(norm1(h))` → `h = h + mlp(norm2(h))` 구조다.
   blk31 출력을 잔차(blk30 출력)·attn 출력·mlp 출력으로 나눠, blk31의 최대 절댓값이
   나오는 좌표에서 각 성분의 값과 **부호**를 본다. 개입 시 baseline의 blk31이 오히려
   커졌던 현상(46,592 → 69,120)이 잔차-분기 상쇄로 설명되는지 확인한다.
3. **RMSNorm 흡수 확인** — 잔차를 c배 했을 때 norm1/norm2 출력이 실제로 불변인지 잰다.
   개입 실험이 "잔차 운반분만 조절한다"는 전제가 성립하는지 검증한다.

학습 불필요. 실행:
    python scripts/probe_blk31_decompose.py --n_images 30
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images
from profile_vision_activations import (
    DRIVELM_VAL_JSON, BASE_3B, MAX_PIXELS, MIN_PIXELS, build_pixel_cache,
    get_visual, percentile,
)

VARIANTS = {
    "baseline_v2": "checkpoints/student_baseline_v2/epoch_1",
    "kd_only_v4":  "checkpoints/student_kd_only_v4/epoch_1",
    "temporal":    "checkpoints/student_temporal/epoch_1",
    "spatial":     "checkpoints/student_spatial/epoch_1",
    "full":        "checkpoints/student_full/epoch_1",
}
SCALES = [("무개입", 1.0), ("잔차 1/34배", 1.0 / 34), ("잔차 34배", 34.0)]


def t_of(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def run(model, cache, scale):
    visual = get_visual(model)
    blk31 = visual.blocks[31]
    rec = {}

    def h30(_m, _i, out):
        t = t_of(out)
        if scale != 1.0:
            t = t * scale
        rec["resid"] = t.detach()
        return t if not isinstance(out, tuple) else (t,) + out[1:]

    def grab(key):
        def fn(_m, _i, out):
            rec[key] = t_of(out).detach()
        return fn

    hs = [visual.blocks[30].register_forward_hook(h30),
          blk31.attn.register_forward_hook(grab("attn")),
          blk31.mlp.register_forward_hook(grab("mlp")),
          blk31.norm1.register_forward_hook(grab("norm1")),
          blk31.norm2.register_forward_hook(grab("norm2")),
          blk31.register_forward_hook(grab("out"))]

    acc = {k: [] for k in ("resid", "attn", "mlp", "out", "norm1", "norm2",
                           "at_resid", "at_attn", "at_mlp", "at_out")}
    for pv, grid in cache:
        rec.clear()
        visual(pv.to("cuda", torch.bfloat16), grid.to("cuda"))
        o = rec["out"].float()
        for k in ("resid", "attn", "mlp", "out", "norm1", "norm2"):
            acc[k].append(rec[k].float().abs().max().item())
        # blk31 최대 절댓값 좌표에서 각 성분의 부호 있는 값
        idx = o.abs().flatten().argmax()
        flat = lambda x: x.float().flatten()[idx].item()
        acc["at_out"].append(flat(rec["out"]))
        acc["at_resid"].append(flat(rec["resid"]))
        acc["at_attn"].append(flat(rec["attn"]))
        acc["at_mlp"].append(flat(rec["mlp"]))

    for x in hs:
        x.remove()
    med = lambda v: percentile(sorted(v), 0.5)
    return {k: med(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=30)
    ap.add_argument("--out", default="eval_results/blk31_decompose.json")
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.n_images]
    proc = AutoProcessor.from_pretrained(BASE_3B, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    cache = build_pixel_cache(proc, paths, 16)

    out = {"n_images": len(paths), "results": {}}
    for name, path in VARIANTS.items():
        print(f"\n[{name}]", flush=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_3B, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, path).to("cuda").eval()
        for label, sc in SCALES:
            r = run(model, cache, sc)
            out["results"][f"{name}|{label}"] = r
            print(f"  {label:12s} resid={r['resid']:>10,.0f} attn={r['attn']:>8,.0f} "
                  f"mlp={r['mlp']:>10,.0f} → blk31={r['out']:>10,.0f}   "
                  f"norm1={r['norm1']:>7.2f} norm2={r['norm2']:>7.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n" + "=" * 96)
    print("1) 개입 강건성 — 잔차를 34배 줄였을 때 blk31 유지율")
    print(f"{'변형':14s}{'무개입':>12}{'1/34배':>12}{'유지율':>9}{'34배':>12}{'증가율':>9}")
    print("-" * 96)
    for n in VARIANTS:
        a = out["results"][f"{n}|무개입"]["out"]
        b = out["results"][f"{n}|잔차 1/34배"]["out"]
        c = out["results"][f"{n}|잔차 34배"]["out"]
        print(f"{n:14s}{a:>12,.0f}{b:>12,.0f}{b/a*100:>8.0f}%{c:>12,.0f}{c/a:>8.1f}x")

    print("\n2) 성분 분해 — blk31 최댓값 좌표에서 각 성분의 부호 있는 값 (무개입)")
    print(f"{'변형':14s}{'잔차':>12}{'attn':>10}{'mlp':>12}{'합(blk31)':>12}")
    print("-" * 96)
    for n in VARIANTS:
        r = out["results"][f"{n}|무개입"]
        print(f"{n:14s}{r['at_resid']:>12,.0f}{r['at_attn']:>10,.0f}"
              f"{r['at_mlp']:>12,.0f}{r['at_out']:>12,.0f}")

    print("\n3) RMSNorm 흡수 — 잔차를 34배 했을 때 norm 출력이 불변인가")
    print(f"{'변형':14s}{'norm1 무개입':>14}{'norm1 34배':>13}{'norm2 무개입':>14}{'norm2 34배':>13}")
    print("-" * 96)
    for n in VARIANTS:
        a = out["results"][f"{n}|무개입"]
        b = out["results"][f"{n}|잔차 34배"]
        print(f"{n:14s}{a['norm1']:>14.2f}{b['norm1']:>13.2f}{a['norm2']:>14.2f}{b['norm2']:>13.2f}")
    print("=" * 96)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
