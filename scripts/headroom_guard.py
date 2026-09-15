"""
headroom_guard — 저정밀 배포 전 표현 범위 여유를 진단하고 교정한다.

**왜 이 도구인가.** 본 연구의 주장은 "저정밀 배포의 성패를 가르는 값이 순전파 한 번으로
계산되는데도 표준 평가에 그 값이 없다"는 것이다. 그렇게 말하면서 재는 도구를 주지 않으면
주장이 불완전하다. 이 스크립트는 논문의 진단을 **실행 가능한 패스**로 묶는다.

  diagnose : 층별 여유를 재고 위험 층을 자동 검출한다.
  fix      : 위험 층에 필요한 스케일을 자동 산출해 가중치를 교정하고, 교정 전후를 재측정한다.

**두 dtype을 나눠 쓴다 (설계 원칙).** 활성 **크기**는 넘치지 않는 dtype(bf16)에서 재고,
**붕괴**는 목표 dtype(fp16)에서 잰다. 목표 dtype에서 크기를 재면 이미 발산한 이미지의
값이 inf/NaN이라 참값을 알 수 없고, 필요한 축소량을 **과소평가**한다
(실측: 10장 중 8장이 발산한 상태로 풀어 목표 2.0배 대신 1.74배에 그쳤다).
여유의 정의 `format_max ÷ max|activation|`에서 분모는 **참값**이어야 한다.

**정직하게 짚을 한계.** 교정은 **완전 등가가 아니다.** 게이트형 MLP에서 곱이 만들어지기
전에 줄이면 오버플로는 사라지지만, 잔차 분기와 MLP 분기의 비율이 바뀐다. 뒤따르는
정규화가 대부분 흡수하나 전부는 아니다(실측: 정확도 −0.59%p). 따라서 이 도구는
**교정을 권하지 않고, 비용과 함께 보고**한다. 채택 여부는 사용자가 정한다.

실행:
    python scripts/headroom_guard.py diagnose --checkpoint checkpoints/student_baseline_v2/epoch_1
    python scripts/headroom_guard.py fix --checkpoint checkpoints/student_baseline_v2/epoch_1 \
        --target_headroom 2.0 --verify
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images

BASE_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
FORMAT_MAX = {"float16": 65504.0, "bfloat16": 3.3895e38, "float32": 3.4028e38}

# 게이트형 MLP의 '곱 이전' 선형층과 '출력' 선형층 이름 후보.
# 계열마다 다르므로 짝으로 둔다. (곱 이전이 둘이면 게이트형, 하나면 단순 MLP)
MLP_PATTERNS = [
    (("gate_proj", "up_proj"), "down_proj"),   # Qwen2.5-VL, LLaMA 계열
    (("fc1",), "fc2"),                          # Qwen2-VL, CLIP 계열
]


# ---------------------------------------------------------------- 모델·타워
def load_tower(checkpoint, base, dtype):
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=getattr(torch, dtype), attn_implementation="eager")
    if checkpoint and checkpoint.lower() != "none":
        m = PeftModel.from_pretrained(m, checkpoint).merge_and_unload()
    return m.visual.eval().to("cuda")


def find_mlp_parts(block):
    """블록의 MLP에서 (곱 이전 선형층들, 출력 선형층)을 찾는다."""
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        return None, None
    for pre_names, out_name in MLP_PATTERNS:
        if all(hasattr(mlp, n) for n in pre_names) and hasattr(mlp, out_name):
            return [getattr(mlp, n) for n in pre_names], getattr(mlp, out_name)
    return None, None


# ---------------------------------------------------------------- 측정
@torch.no_grad()
def profile(tower, cache, dtype, fmax_dtype=None):
    """블록별 max|activation|과 출력 NaN 비율을 잰다.

    `fmax_dtype`을 주면 여유 계산의 분자를 그 dtype의 상한으로 쓴다
    (크기는 bf16에서 재고 여유는 fp16 기준으로 보고할 때).
    """
    blocks = tower.blocks
    rec = {i: [] for i in range(len(blocks))}
    hs = []

    def mk(i):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                v = t.detach().float().abs().max().item()
                rec[i].append(v)
        return fn
    for i, b in enumerate(blocks):
        hs.append(b.register_forward_hook(mk(i)))

    nan = 0
    for pv, grid in cache:
        out = tower(pv, grid)
        if not torch.isfinite(out).all():
            nan += 1
    for h in hs:
        h.remove()

    import statistics as st
    fmax = FORMAT_MAX[fmax_dtype or dtype]
    per_block = []
    for i in range(len(blocks)):
        vals = [v for v in rec[i] if math.isfinite(v)]
        p50 = st.median(vals) if vals else float("nan")
        per_block.append({"block": i, "max_p50": p50,
                          "headroom": (fmax / p50) if p50 and math.isfinite(p50) else None,
                          "n_finite": len(vals)})
    return per_block, nan / len(cache)


def build_cache(processor, paths, dtype):
    cache = []
    for p in paths:
        enc = processor.image_processor(images=[Image.open(p).convert("RGB")],
                                        return_tensors="pt")
        cache.append((enc["pixel_values"].to("cuda", getattr(torch, dtype)),
                      enc["image_grid_thw"].to("cuda")))
    return cache


# ---------------------------------------------------------------- 교정
@torch.no_grad()
def measure_branches(tower, cache, block_idx, dtype):
    """블록 출력의 최대 지점에서 **잔차 분기와 MLP 분기를 분리**해 잰다.

    **왜 필요한가.** 블록 출력은 `잔차 + MLP(...)`다. MLP 분기만 f배로 줄여도
    잔차는 그대로 남으므로, 목표 여유를 단순히 `target/현재`로 계산하면 맞지 않는다
    (실측: 목표 2.0배를 지정했는데 1.70배가 나왔다). 두 분기를 분리해 재고
    `|r + f·m| ≤ target_max`를 f에 대해 풀어야 한다.
    """
    blk = tower.blocks[block_idx]
    buf = {}

    def h_mlp(_m, _i, out):
        buf["mlp"] = (out[0] if isinstance(out, tuple) else out).detach()

    def h_blk(_m, _i, out):
        buf["out"] = (out[0] if isinstance(out, tuple) else out).detach()

    hs = [blk.mlp.register_forward_hook(h_mlp), blk.register_forward_hook(h_blk)]
    pairs = []
    for pv, grid in cache:
        buf.clear()
        tower(pv, grid)
        if "mlp" not in buf or "out" not in buf:
            continue
        o, m = buf["out"].float(), buf["mlp"].float()
        if not (torch.isfinite(o).all() and torch.isfinite(m).all()):
            # 이미 발산한 이미지는 분해가 무의미하므로 MLP 쪽만 본다
            mm = m[torch.isfinite(m)]
            if mm.numel():
                pairs.append((0.0, float(mm.abs().max())))
            continue
        idx = int(o.abs().argmax())
        ov, mv = float(o.flatten()[idx]), float(m.flatten()[idx])
        pairs.append((ov - mv, mv))          # (잔차, MLP)
    for h in hs:
        h.remove()
    return pairs


def solve_factor(pairs, target_max):
    """|r + f·m| <= target_max 를 만족하는 최대 f를 구한다 (모든 이미지에 대해)."""
    fs, infeasible = [], 0
    for r, m in pairs:
        if abs(m) < 1e-6:
            continue
        if abs(r) >= target_max:
            infeasible += 1          # 잔차만으로 이미 목표를 넘는다
            continue
        fs.append((target_max - abs(r)) / abs(m))
    if not fs:
        return None, infeasible, 0
    return max(1e-3, min(min(fs), 1.0)), infeasible, len(fs)


def plan_fix(per_block, target, tower=None, cache=None, dtype="float16"):
    """여유가 target 미만인 블록과 필요한 MLP 축소 배수를 산출한다.

    tower/cache가 주어지면 잔차를 분리해 정확히 풀고, 없으면 근사값을 쓴다.
    """
    fmax = FORMAT_MAX[dtype]
    target_max = fmax / target
    plan = []
    for b in per_block:
        h = b["headroom"]
        if h is None or h >= target:
            continue
        item = {"block": b["block"], "headroom": h}
        if tower is not None and cache is not None:
            pairs = measure_branches(tower, cache, b["block"], dtype)
            f, infeas, n = solve_factor(pairs, target_max)
            if f is None:
                item.update({"scale": target / h, "note":
                             "분기 분리 실패 — 근사값 사용"})
            else:
                item.update({"scale": 1.0 / f, "factor_mlp": f,
                             "n_solved": n, "n_infeasible": infeas})
                if infeas:
                    item["warn"] = (f"{infeas}장에서 **잔차만으로 목표를 넘는다** — "
                                    f"MLP 축소만으로는 목표 여유에 도달할 수 없다")
        else:
            item["scale"] = target / h
        plan.append(item)
    return plan


def apply_fix(tower, plan):
    """곱이 만들어지기 **전에** 줄인다. 게이트형이면 각 선형층을 1/sqrt(scale)로.

    MLP에 bias가 있으므로(Qwen2.5-VL 비전 MLP는 bias=True) **weight와 bias를 함께**
    나눠야 순전파 훅과 등가다. weight만 나누면 bias가 남아 다른 연산이 된다.
    """
    applied = []
    for item in plan:
        blk = tower.blocks[item["block"]]
        pres, out_lin = find_mlp_parts(blk)
        if pres is None:
            applied.append({**item, "skipped": "MLP 구조를 인식하지 못함"})
            continue
        f = (1.0 / item["scale"]) ** (1.0 / len(pres))   # 곱이 1/scale이 되도록
        with torch.no_grad():
            for lin in pres:
                lin.weight.mul_(f)
                if getattr(lin, "bias", None) is not None:
                    lin.bias.mul_(f)
        applied.append({**item, "factor_per_layer": f, "n_layers": len(pres),
                        "has_bias": getattr(pres[0], "bias", None) is not None})
    return applied


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "fix"])
    ap.add_argument("--checkpoint", default="checkpoints/student_baseline_v2/epoch_1",
                    help="'none'이면 사전학습 베이스")
    ap.add_argument("--base", default=BASE_3B)
    ap.add_argument("--dtype", default="float16", choices=list(FORMAT_MAX))
    ap.add_argument("--max_pixels", type=int, default=None,
                    help="미지정이면 **모델 기본 설정**을 쓴다 — 기본 경로에서의 위험을 보려면 그대로 둘 것")
    ap.add_argument("--min_pixels", type=int, default=None)
    ap.add_argument("--image_dir", default=None, help="미지정이면 DriveLM val CAM_FRONT")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--target_headroom", type=float, default=2.0,
                    help="이 값 미만인 블록을 위험으로 보고, fix에서는 이 값까지 끌어올린다")
    ap.add_argument("--verify", action="store_true", help="fix 후 재측정")
    ap.add_argument("--save_weights", default=None, help="교정된 비전 타워 state_dict 저장 경로")
    ap.add_argument("--out", default="eval_results/headroom_guard.json")
    args = ap.parse_args()

    awq_compat.patch()
    from transformers import AutoProcessor

    kw = {}
    if args.max_pixels:
        kw["max_pixels"] = args.max_pixels
    if args.min_pixels:
        kw["min_pixels"] = args.min_pixels
    proc = AutoProcessor.from_pretrained(args.base, **kw)
    ip = proc.image_processor
    print(f"[설정] dtype={args.dtype}  format_max={FORMAT_MAX[args.dtype]:,.0f}")
    print(f"        max_pixels={ip.max_pixels:,}  min_pixels={ip.min_pixels:,}"
          + ("  (모델 기본값)" if not args.max_pixels else ""))

    if args.image_dir:
        paths = sorted(str(p) for p in Path(args.image_dir).glob("*.jpg"))[:args.limit]
    else:
        tok = build_token_to_images(DRIVELM_VAL)
        paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:args.limit]
    print(f"[이미지] {len(paths)}장")

    # 크기는 넘치지 않는 dtype에서, 붕괴는 목표 dtype에서 잰다 (docstring 참조)
    MAG = "bfloat16" if args.dtype == "float16" else args.dtype
    tower = load_tower(args.checkpoint, args.base, MAG)
    cache_mag = build_cache(proc, paths, MAG)
    print(f"[형상] pixel_values={tuple(cache_mag[0][0].shape)}  "
          f"grid={cache_mag[0][1].tolist()}")
    print(f"[측정] 크기={MAG} (참값)  붕괴={args.dtype}\n")

    before, _ = profile(tower, cache_mag, MAG, fmax_dtype=args.dtype)
    risky = plan_fix(before, args.target_headroom,
                     tower=tower, cache=cache_mag, dtype=args.dtype)

    # 붕괴는 목표 dtype으로 갈아끼워 잰다
    def collapse_in_target():
        tower.to(getattr(torch, args.dtype))
        c = build_cache(proc, paths, args.dtype)
        n = 0
        with torch.no_grad():
            for pv, grid in c:
                if not torch.isfinite(tower(pv, grid)).all():
                    n += 1
        tower.to(getattr(torch, MAG))
        return n / len(c)

    nan_before = collapse_in_target()

    print(f"{'blk':>4}{'max|act| p50':>15}{'여유':>9}")
    for b in before:
        if b["headroom"] is None:
            print(f"{b['block']:>4}{'(비유한)':>15}{'—':>9}")
        elif b["headroom"] < args.target_headroom or b["block"] >= len(before) - 3:
            mark = "  ← 위험" if b["headroom"] < args.target_headroom else ""
            print(f"{b['block']:>4}{b['max_p50']:>15,.0f}{b['headroom']:>8.2f}배{mark}")
    print(f"\n[진단] 위험 블록 {len(risky)}개 (기준: 여유 < {args.target_headroom})")
    print(f"       현재 붕괴율 {nan_before*100:.1f}%  ({len(paths)}장)")

    res = {"checkpoint": args.checkpoint, "dtype": args.dtype,
           "max_pixels": ip.max_pixels, "n_images": len(paths),
           "target_headroom": args.target_headroom,
           "before": {"per_block": before, "nan_rate": nan_before},
           "risky_blocks": risky}

    if args.cmd == "fix":
        if not risky:
            print("\n교정할 블록이 없다 — 이 설정에서는 안전하다.")
        else:
            print(f"\n[교정] 곱이 만들어지기 전에 축소한다")
            applied = apply_fix(tower, risky)
            for a in applied:
                if "skipped" in a:
                    print(f"  blk{a['block']}: 건너뜀 — {a['skipped']}")
                else:
                    print(f"  blk{a['block']}: 여유 {a['headroom']:.2f}배 → 목표 "
                          f"{args.target_headroom:.1f}배 | MLP 분기 ×"
                          f"{1/a['scale']:.4f}, 선형층 {a['n_layers']}개를 각 "
                          f"×{a['factor_per_layer']:.6f}"
                          + ("  (bias 포함)" if a["has_bias"] else ""))
                    if a.get("warn"):
                        print(f"    ⚠ {a['warn']}")
            res["applied"] = applied

            if args.verify:
                after, _ = profile(tower, cache_mag, MAG, fmax_dtype=args.dtype)
                nan_after = collapse_in_target()
                last = after[-1]
                print(f"\n[검증] 붕괴율 {nan_before*100:.1f}% → **{nan_after*100:.1f}%**")
                print(f"       마지막 블록 여유 {before[-1]['headroom']:.2f}배 → "
                      f"{last['headroom']:.2f}배")
                res["after"] = {"per_block": after, "nan_rate": nan_after}
                print("\n  ⚠ 이 교정은 **완전 등가가 아니다.** 잔차 분기와 MLP 분기의")
                print("    비율이 바뀌며, 뒤따르는 정규화가 대부분 흡수하나 전부는 아니다.")
                print("    정확도 비용은 별도로 측정할 것 "
                      "(실측 사례: 11,309문항에서 −0.59%p).")

            if args.save_weights:
                torch.save(tower.state_dict(), args.save_weights)
                print(f"\n  교정된 비전 타워 저장: {args.save_weights}")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
