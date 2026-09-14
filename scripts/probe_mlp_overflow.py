"""
blk31 MLP 내부 오버플로 지점 특정 + 사후 수정 재시도
======================================================
사후 리스케일 실험에서 blk31 **출력**을 1/3로 줄여도 fp16 붕괴율이 94.4% → 94.8%로
그대로였다. 오버플로가 hook보다 먼저, 즉 MLP **내부**에서 일어나기 때문으로 보인다.

비전 MLP 구조: `down_proj( act_fn(gate_proj(x)) * up_proj(x) )` (bias 없음).
따라서 후보 지점은 넷이다 — gate_proj 출력, up_proj 출력, 둘의 곱, down_proj 출력.

  (1) **진단**: 각 지점에 hook을 걸고 fp16으로 통과시켜 inf/NaN이 **처음** 나타나는 곳을 센다.
  (2) **수정 재시도**: 진단된 지점보다 **앞에서** 스케일을 줄이면 붕괴를 막을 수 있는가.
      - fixA: gate·up 출력을 각각 1/√s → 곱이 1/s (down_proj 계산 전에 축소)
      - fixB: down_proj 출력만 1/s (이미 실패한 조건, 대조군)

bias가 없어 선형층 출력 스케일링은 입력 스케일링과 등가라, hook 개입이 가중치 편집과 같다.

실행:
    python scripts/probe_mlp_overflow.py --scale 3 --n_images 250
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images
from profile_vision_activations import (
    DRIVELM_VAL_JSON, BASE_3B, build_pixel_cache, get_visual, percentile,
)

CKPT = "checkpoints/student_baseline_v2/epoch_1"
STAGES = ("norm2", "gate", "up", "product", "down", "block")


def t_of(o):
    return o[0] if isinstance(o, tuple) else o


@torch.no_grad()
def run(model, cache, mode, s):
    """mode: none | fixA(gate·up 축소) | fixB(down 출력 축소)"""
    visual = get_visual(model.to(torch.float16))
    blk = visual.blocks[31]
    mlp = blk.mlp
    rec, hs = {}, []

    def grab(k):
        def fn(_m, _i, out):
            rec[k] = t_of(out).detach()
        return fn

    def grab_scale(k, f):
        def fn(_m, _i, out):
            t = t_of(out)
            rec[k] = t.detach()
            if f == 1.0:
                return None
            sc = t * f
            rec[k + "_post"] = sc.detach()
            return sc if not isinstance(out, tuple) else (sc,) + out[1:]
        return fn

    fa = 1.0 / math.sqrt(s) if mode == "fixA" else 1.0
    fb = 1.0 / s if mode == "fixB" else 1.0

    hs.append(blk.norm2.register_forward_hook(grab("norm2")))
    hs.append(mlp.gate_proj.register_forward_hook(grab_scale("gate", fa)))
    hs.append(mlp.up_proj.register_forward_hook(grab_scale("up", fa)))
    hs.append(mlp.down_proj.register_forward_pre_hook(
        lambda _m, inp: rec.__setitem__("product", inp[0].detach())))
    hs.append(mlp.down_proj.register_forward_hook(grab_scale("down", fb)))
    hs.append(blk.register_forward_hook(grab("block")))

    first_bad = {k: 0 for k in STAGES}
    maxes = {k: [] for k in STAGES}
    n_bad = 0
    for pv, grid in cache:
        rec.clear()
        visual(pv.to("cuda", torch.float16), grid.to("cuda"))
        bad_at = None
        for k in STAGES:
            t = rec.get(k)
            if t is None:
                continue
            f = t.float()
            if torch.isnan(f).any() or torch.isinf(f).any():
                if bad_at is None:
                    bad_at = k
            else:
                maxes[k].append(f.abs().max().item())
        if bad_at:
            first_bad[bad_at] += 1
            n_bad += 1

    for h in hs:
        h.remove()
    model.to(torch.bfloat16)
    med = lambda v: percentile(sorted(v), 0.5) if v else None
    return {"mode": mode, "collapse_rate": n_bad / len(cache), "n_bad": n_bad,
            "n": len(cache), "first_inf_stage": first_bad,
            "stage_max_p50": {k: med(v) for k, v in maxes.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--n_images", type=int, default=250)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--out", default="eval_results/mlp_overflow.json")
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.n_images]
    proc = AutoProcessor.from_pretrained(BASE_3B, max_pixels=args.max_pixels, min_pixels=3136)
    cache = build_pixel_cache(proc, paths, 16)

    print("[모델 로드]", flush=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_3B, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, CKPT).to("cuda").eval()

    out = {"checkpoint": CKPT, "scale": args.scale, "max_pixels": args.max_pixels, "runs": {}}
    for mode in ("none", "fixB", "fixA"):
        r = run(model, cache, mode, args.scale)
        out["runs"][mode] = r
        stages = {k: v for k, v in r["first_inf_stage"].items() if v}
        print(f"\n[{mode}] 붕괴 {r['n_bad']}/{r['n']} = {r['collapse_rate']*100:.1f}%")
        print(f"    최초 inf 지점: {stages if stages else '없음'}")
        print("    단계별 max|x| p50: " + "  ".join(
            f"{k}={v:,.0f}" for k, v in r["stage_max_p50"].items() if v is not None), flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + "=" * 74)
    print(f"{'조건':8s}{'붕괴율':>10}   최초 inf 지점")
    print("-" * 74)
    for m in ("none", "fixB", "fixA"):
        r = out["runs"][m]
        st = {k: v for k, v in r["first_inf_stage"].items() if v}
        print(f"{m:8s}{r['collapse_rate']*100:>9.1f}%   {st if st else '없음'}")
    print("=" * 74)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
