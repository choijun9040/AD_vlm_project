"""
fixA의 두 구현이 같은 결과를 내는지 확인한다 (논문 7.4 전제).

`probe_posthoc_rescale.py`의 fixA는 gate/up의 **출력**을 스케일하는 순전파 훅이다.
ONNX로 내보내려면 이를 **가중치 편집**으로 옮겨야 하는데, 비전 MLP에는 bias가 있으므로
(`Qwen2_5_VLMLP(config, bias=True)`) weight와 bias를 **함께** 나눠야 등가다.
weight만 나누면 bias가 남아 다른 연산이 된다.

이 스크립트는 세 조건의 fp16 상실률을 같은 250장에서 비교한다:
    무개입 / 훅 fixA / 가중치 편집 fixA(weight+bias)
그리고 대조로 **weight만** 편집한 잘못된 판본도 함께 잰다 — 다르게 나와야 정상이다.

실행:
    python scripts/check_fixa_weight_equiv.py --limit 250
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
CKPT = "checkpoints/student_baseline_v2/epoch_1"
VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"


def hook_fixa(tower, scale):
    """gate/up의 출력을 1/sqrt(scale)배 — 원래 실험과 동일한 개입."""
    f = 1.0 / math.sqrt(scale)
    mlp = tower.blocks[-1].mlp
    return [mlp.gate_proj.register_forward_hook(lambda m, i, o: o * f),
            mlp.up_proj.register_forward_hook(lambda m, i, o: o * f)]


def edit_weights(tower, scale, with_bias):
    """가중치를 직접 편집하고 되돌릴 수 있는 복원 함수를 반환한다."""
    f = 1.0 / math.sqrt(scale)
    mlp = tower.blocks[-1].mlp
    saved = []
    with torch.no_grad():
        for lin in (mlp.gate_proj, mlp.up_proj):
            saved.append((lin, lin.weight.detach().clone(),
                          None if lin.bias is None else lin.bias.detach().clone()))
            lin.weight.mul_(f)
            if with_bias and lin.bias is not None:
                lin.bias.mul_(f)

    def restore():
        with torch.no_grad():
            for lin, w, b in saved:
                lin.weight.copy_(w)
                if b is not None:
                    lin.bias.copy_(b)
    return restore


def collapse(tower, cache, flags=None):
    """flags: 리스트를 주면 이미지별 상실(비유한 출력) 여부를 덧붙인다 — 집계에는 영향 없음."""
    bad = 0
    with torch.no_grad():
        for pv, grid in cache:
            out = tower(pv, grid)
            nf = not torch.isfinite(out).all()
            if nf:
                bad += 1
            if flags is not None:
                flags.append(bool(nf))
    return bad / len(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--offset", type=int, default=0, help="이미지 시작 위치 (표준 순서 기준)")
    ap.add_argument("--out", default=None,
                    help="출력하는 집계(조건별 상실률·판정)를 JSON으로 저장. 미지정이면 저장하지 않는다(예전 동작)")
    ap.add_argument("--per_image", default=None,
                    help="조건·이미지별 상실 여부와 장면 토큰을 jsonl로 저장('w'). 장면 군집 부트스트랩용")
    args = ap.parse_args()

    awq_compat.patch()
    print(f"[로드] {CKPT}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, CKPT).merge_and_unload()
    tower = model.visual.eval().to("cuda").half()

    mlp = tower.blocks[-1].mlp
    print(f"  bias 존재: gate={mlp.gate_proj.bias is not None} "
          f"up={mlp.up_proj.bias is not None} down={mlp.down_proj.bias is not None}")

    proc = AutoProcessor.from_pretrained(BASE, max_pixels=args.max_pixels,
                                         min_pixels=args.min_pixels)
    tok = build_token_to_images(VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][args.offset:args.offset + args.limit]
    print(f"[캐시] {len(paths)}장 전처리")
    cache = []
    for p in paths:
        enc = proc.image_processor(images=[Image.open(p).convert("RGB")],
                                   return_tensors="pt")
        cache.append((enc["pixel_values"].to("cuda", torch.float16),
                      enc["image_grid_thw"].to("cuda")))

    want = args.per_image is not None
    flags = {k: ([] if want else None) for k in ("none", "hook_fixa", "weight_bias_fixa", "weight_only_fixa")}
    print(f"\n{'조건':<34}{'상실률':>10}")
    r = collapse(tower, cache, flags["none"])
    print(f"{'무개입':<34}{r*100:>9.1f}%")

    hs = hook_fixa(tower, args.scale)
    r_hook = collapse(tower, cache, flags["hook_fixa"])
    for h in hs:
        h.remove()
    print(f"{'훅 fixA (원래 실험)':<34}{r_hook*100:>9.1f}%")

    restore = edit_weights(tower, args.scale, with_bias=True)
    r_wb = collapse(tower, cache, flags["weight_bias_fixa"])
    restore()
    print(f"{'가중치 fixA (weight+bias)':<34}{r_wb*100:>9.1f}%   ← ONNX로 내보낸 것")

    restore = edit_weights(tower, args.scale, with_bias=False)
    r_w = collapse(tower, cache, flags["weight_only_fixa"])
    restore()
    print(f"{'가중치 fixA (weight만, 잘못된 판본)':<34}{r_w*100:>9.1f}%")

    print()
    if abs(r_hook - r_wb) < 1e-9:
        print("판정: weight+bias 편집이 훅과 **정확히 일치** — ONNX 판본이 유효하다")
    else:
        print(f"판정: 불일치 ({r_hook*100:.1f}% 대 {r_wb*100:.1f}%) — "
              f"등가성 가정을 다시 봐야 한다")
    if abs(r_w - r_wb) > 1e-9:
        print(f"      weight만 편집한 판본은 {r_w*100:.1f}%로 다르다 — bias 정정이 필요했음이 확인된다")
    else:
        print(f"      weight만 편집해도 같다 — 이 스케일에서는 bias 기여가 무시할 수준이다")

    if args.out:
        res = {"checkpoint": CKPT, "scale": args.scale, "max_pixels": args.max_pixels,
               "min_pixels": args.min_pixels, "n_images": len(paths), "offset": args.offset,
               "collapse_rate": {"none": r, "hook_fixa": r_hook,
                                 "weight_bias_fixa": r_wb, "weight_only_fixa": r_w},
               "hook_equals_weight_bias": abs(r_hook - r_wb) < 1e-9,
               "weight_only_differs": abs(r_w - r_wb) > 1e-9}
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"저장: {args.out}")
    if want:
        from scene_map import image_to_scene
        sc = image_to_scene()
        with open(args.per_image, "w") as f:
            for cond, fl in flags.items():
                for p, nf in zip(paths, fl):
                    nm = Path(p).name
                    f.write(json.dumps({"condition": cond, "checkpoint": CKPT, "scale": args.scale,
                                        "max_pixels": args.max_pixels, "image": nm,
                                        "scene": sc.get(nm), "nonfinite_fp16": nf}) + "\n")
        print(f"이미지별: {args.per_image}")


if __name__ == "__main__":
    main()
