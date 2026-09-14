"""
잔차 개입의 간접 경로 규명 — norm2 출력의 '방향' 변화
========================================================
`probe_blk31_decompose.py`에서 잔차를 34배 했을 때 norm2의 **최댓값**은 거의 그대로인데
(31.00 → 31.25) blk31 출력은 5.7배 변했다. 이 불일치를 설명하는 가설:

  비전 블록은 `h1 = h + attn(norm1(h))`, `out = h1 + mlp(norm2(h1))` 구조다.
  norm1의 입력은 **순수 잔차**이므로 스케일이 RMSNorm에 완전히 흡수된다(방향 불변).
  그러나 norm2의 입력은 `잔차 + attn출력`이라, **잔차만** 스케일하면 두 항의 비율이
  바뀐다. RMSNorm은 크기를 고정하지만 **방향은 고정하지 않으므로**, norm2 출력의
  방향이 달라지고 MLP가 다른 입력을 받는다.

검증: 무개입과 스케일 조건에서 같은 이미지의 norm1·norm2 출력 사이 **코사인 유사도**를
잰다. 가설이 맞으면 norm1은 ≈1.000, norm2는 뚜렷하게 낮아야 한다.

실행:
    python scripts/probe_norm2_direction.py --n_images 20
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
    "full":        "checkpoints/student_full/epoch_1",
}
SCALES = [1.0 / 34, 34.0]


def t_of(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def capture(model, cache, scale):
    """조건별로 norm1·norm2·mlp 출력과 잔차/attn 비율을 이미지별로 모은다."""
    visual = get_visual(model)
    blk = visual.blocks[31]
    rec, store = {}, []

    def h30(_m, _i, out):
        t = t_of(out)
        if scale != 1.0:
            t = t * scale
        rec["resid"] = t.detach()
        return t if not isinstance(out, tuple) else (t,) + out[1:]

    def grab(k):
        def fn(_m, _i, out):
            rec[k] = t_of(out).detach()
        return fn

    hs = [visual.blocks[30].register_forward_hook(h30),
          blk.norm1.register_forward_hook(grab("norm1")),
          blk.norm2.register_forward_hook(grab("norm2")),
          blk.attn.register_forward_hook(grab("attn")),
          blk.mlp.register_forward_hook(grab("mlp"))]

    for pv, grid in cache:
        rec.clear()
        visual(pv.to("cuda", torch.bfloat16), grid.to("cuda"))
        store.append({k: rec[k].float().cpu() for k in
                      ("norm1", "norm2", "mlp", "resid", "attn")})
    for x in hs:
        x.remove()
    return store


def med_cos(a_list, b_list, key):
    """토큰별 코사인 유사도의 중앙값을 이미지마다 구해 다시 중앙값."""
    per_image = []
    for a, b in zip(a_list, b_list):
        c = F.cosine_similarity(a[key], b[key], dim=-1)
        per_image.append(float(c.median()))
    return percentile(sorted(per_image), 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=20)
    ap.add_argument("--out", default="eval_results/norm2_direction.json")
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.n_images]
    proc = AutoProcessor.from_pretrained(BASE_3B, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    cache = build_pixel_cache(proc, paths, 16)

    out = {"n_images": len(paths), "results": {}}
    for name, path in VARIANTS.items():
        print(f"\n[{name}] 로드", flush=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_3B, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, path).to("cuda").eval()

        ref = capture(model, cache, 1.0)
        # 무개입에서 잔차 대 attn 크기 비 — norm2 입력의 구성비
        ratio = percentile(sorted(float(r["resid"].abs().max() / r["attn"].abs().max().clamp_min(1e-9))
                                  for r in ref), 0.5)
        print(f"  무개입 잔차/attn 최댓값 비 = {ratio:.1f}", flush=True)

        for sc in SCALES:
            cur = capture(model, cache, sc)
            r = {"cos_norm1": med_cos(ref, cur, "norm1"),
                 "cos_norm2": med_cos(ref, cur, "norm2"),
                 "cos_mlp":   med_cos(ref, cur, "mlp"),
                 "resid_attn_ratio": ratio}
            out["results"][f"{name}|x{sc:.4g}"] = r
            print(f"  스케일 x{sc:<7.4g} cos(norm1)={r['cos_norm1']:.5f}  "
                  f"cos(norm2)={r['cos_norm2']:.5f}  cos(mlp)={r['cos_mlp']:.5f}", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n" + "=" * 78)
    print("가설: norm1은 순수 잔차 입력이라 스케일이 완전 흡수(cos≈1),")
    print("      norm2는 잔차+attn 혼합이라 비율 변화로 **방향**이 바뀐다(cos<1).")
    print("=" * 78)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
