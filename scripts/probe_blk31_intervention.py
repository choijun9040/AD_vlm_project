"""
blk31 massive activation의 출처 — 개입 실험
=============================================
관찰: 비전 인코더의 최대 활성은 블록 24~30에서 완전히 정체하다가 **마지막 블록 하나가
전체 크기의 대부분을 만들어낸다**(baseline +42,080 / align +17,679). 또 중간층에서
33.6배였던 두 모델의 격차가 blk31에서 2.8배로 줄어든다.

질문: blk31의 크기는 (a) 앞 블록에서 잔차로 **운반된** 것인가, (b) 마지막 블록이
**자체 생성**한 것인가?  (a)라면 `L_align`의 fp16 이득은 중간층 이상치 억제의 하류
결과이고, (b)라면 별개 현상이다.

방법: blk30 출력에 forward hook으로 스칼라를 곱해 잔차 운반분만 인위적으로 조절하고
blk31 출력을 관찰한다. 비전 블록은 잔차 + RMSNorm 구조라 분기(norm 통과)는 스케일에
둔감하고 잔차만 곱해지므로, 이 개입은 정확히 "운반분 대 생성분"을 가른다.

  - baseline_v2의 blk30을 1/33.6로 축소 → blk31이 여전히 크면 **자체 생성**
  - full의 blk30을 33.6배로 확대 → blk31이 그대로면 **자체 생성** (대칭 확인)

학습 불필요. 실행:
    python scripts/probe_blk31_intervention.py --n_images 30
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images
from profile_vision_activations import (
    DRIVELM_VAL_JSON, BASE_3B, MAX_PIXELS, MIN_PIXELS, build_pixel_cache,
    get_visual, percentile,
)

CKPT = {
    "baseline_v2": "checkpoints/student_baseline_v2/epoch_1",
    "full":        "checkpoints/student_full/epoch_1",
}
# 중간층 격차 — blk30에서 baseline 8,096 대 full 241
GAP = 8096 / 241.0


def as_tensor(out):
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def run(model, cache, scale):
    """blk30 출력에 scale을 곱한 뒤 blk30/blk31 최대 활성을 잰다."""
    visual = get_visual(model)
    rec = {}

    def hook30(_m, _i, out):
        t = as_tensor(out)
        rec["blk30_pre"] = t.detach()
        if scale == 1.0:
            return None
        scaled = t * scale
        rec["blk30_post"] = scaled.detach()
        return scaled if not isinstance(out, tuple) else (scaled,) + out[1:]

    def hook31(_m, _i, out):
        rec["blk31"] = as_tensor(out).detach()

    h = [visual.blocks[30].register_forward_hook(hook30),
         visual.blocks[31].register_forward_hook(hook31)]

    pre, post, b31 = [], [], []
    for pv, grid in cache:
        rec.clear()
        visual(pv.to("cuda", torch.bfloat16), grid.to("cuda"))
        pre.append(rec["blk30_pre"].float().abs().max().item())
        post.append(rec.get("blk30_post", rec["blk30_pre"]).float().abs().max().item())
        b31.append(rec["blk31"].float().abs().max().item())

    for x in h:
        x.remove()
    med = lambda v: percentile(sorted(v), 0.5)
    return {"blk30_pre": med(pre), "blk30_post": med(post), "blk31": med(b31)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_images", type=int, default=30)
    ap.add_argument("--out", default="eval_results/blk31_intervention.json")
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.n_images]
    proc = AutoProcessor.from_pretrained(BASE_3B, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    cache = build_pixel_cache(proc, paths, 16)

    results = {"gap": GAP, "n_images": len(paths), "conditions": {}}
    for name, path in CKPT.items():
        print(f"\n[{name}] 로드", flush=True)
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_3B, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, path).to("cuda").eval()

        conds = [("무개입", 1.0)]
        if name == "baseline_v2":
            conds.append((f"blk30을 1/{GAP:.1f}배 축소", 1.0 / GAP))
        else:
            conds.append((f"blk30을 {GAP:.1f}배 확대", GAP))

        for label, sc in conds:
            r = run(model, cache, sc)
            results["conditions"][f"{name} | {label}"] = r
            print(f"  {label:24s} blk30 {r['blk30_pre']:>9,.0f} → {r['blk30_post']:>9,.0f}"
                  f"   blk31 = {r['blk31']:>10,.0f}", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    base_no = results["conditions"]["baseline_v2 | 무개입"]["blk31"]
    base_sc = [v for k, v in results["conditions"].items() if k.startswith("baseline_v2 | blk30")][0]["blk31"]
    full_no = results["conditions"]["full | 무개입"]["blk31"]
    full_sc = [v for k, v in results["conditions"].items() if k.startswith("full | blk30")][0]["blk31"]

    print("\n" + "=" * 76)
    print("해석")
    print(f"  baseline: 잔차를 {GAP:.0f}배 줄여도 blk31 {base_no:,.0f} → {base_sc:,.0f} "
          f"({base_sc/base_no*100:.0f}% 유지)")
    print(f"  full    : 잔차를 {GAP:.0f}배 늘려도 blk31 {full_no:,.0f} → {full_sc:,.0f} "
          f"({full_sc/full_no:.1f}배)")
    print("  → 유지율이 높을수록 blk31은 마지막 블록의 '자체 생성'이며 중간층과 독립이다.")
    print("=" * 76)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
