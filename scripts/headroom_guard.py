"""
headroom_guard — 저정밀 배포 전 표현 범위 여유를 진단하고 교정한다.

**왜 이 도구인가.** 본 연구의 주장은 "저정밀 배포의 성패를 가르는 값이 순전파 한 번으로
계산되는데도 표준 평가에 그 값이 없다"는 것이다. 그렇게 말하면서 재는 도구를 주지 않으면
주장이 불완전하다. 이 스크립트는 논문의 진단을 **실행 가능한 패스**로 묶는다.

  diagnose : 층별 여유를 재고 위험 층을 자동 검출한다.
  fix      : 위험 층에 필요한 스케일을 자동 산출해 가중치를 교정하고, 교정 전후를 재측정한다.

**형식부터 검토하게 한다 (설계 원칙 1).** 여유는 **형식마다 다른 값**이다 —
같은 활성 74,752가 fp16(한계 65,504)에서는 넘치고 bf16(한계 3.4e38)에서는 한참 남는다.
따라서 도구는 목표 dtype만 보지 않고 **모든 형식의 여유를 함께 보고**하고, 더 안전한
형식이 있으면 **가중치를 고치기 전에 그쪽을 먼저 검토하라고 권고**한다. 가장 명백한
대안을 제시하지 않으면 도구가 정직하지 않다.

**두 dtype을 나눠 쓴다 (설계 원칙 2).** 활성 **크기**는 넘치지 않는 dtype(bf16)에서 재고,
**붕괴**는 목표 dtype(fp16)에서 잰다. 목표 dtype에서 크기를 재면 이미 발산한 이미지의
값이 inf/NaN이라 참값을 알 수 없고, 필요한 축소량을 **과소평가**한다
(실측: 10장 중 8장이 발산한 상태로 풀어 목표 2.0배 대신 1.74배에 그쳤다).
여유의 정의 `format_max ÷ max|activation|`에서 분모는 **참값**이어야 한다.

**정직하게 짚을 한계.** 교정은 **완전 등가가 아니다.** 게이트형 MLP에서 곱이 만들어지기
전에 줄이면 오버플로는 사라지지만, 잔차 분기와 MLP 분기의 비율이 바뀐다. 뒤따르는
정규화가 대부분 흡수하나 전부는 아니다. 다만 **배포 조건에서 잰 비용은 측정 한계
이하**였다 — 원본 해상도·fp16·1,029문항에서 무교정 bf16 기준선 대비 +0.78%p,
McNemar p=0.34(2026-09-15). 이전 판이 인용하던 −0.59%p는 **평가 해상도·bf16**,
즉 오버플로가 없어 교정이 필요 없는 조건에서 잰 값이었다. 따라서 이 도구는
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

# 초과는 없으나 여유가 얇은 구간의 경계. 실측 전이가 최악 여유 1.00~1.05이고
# 한 모델 안 이미지별 편차가 15~20%이므로, 표본 밖 이미지를 위해 1.25배를 둔다.
# (게이트 보정 2026-09-15 — eval_results/gate_calibration.json)
MARGINAL_HEADROOM = 1.25

# 게이트형 MLP의 '곱 이전' 선형층과 '출력' 선형층 이름 후보.
# 계열마다 다르므로 짝으로 둔다. (곱 이전이 둘이면 게이트형, 하나면 단순 MLP)
MLP_PATTERNS = [
    (("gate_proj", "up_proj"), "down_proj"),   # Qwen2.5-VL, LLaMA 계열
    (("fc1",), "fc2"),                          # Qwen2-VL, CLIP 계열
]


# ---------------------------------------------------------------- 모델·타워
def load_tower(checkpoint, base, dtype):
    """우리 체크포인트(LoRA 병합) 경로."""
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base, torch_dtype=getattr(torch, dtype), attn_implementation="eager")
    if checkpoint and checkpoint.lower() != "none":
        m = PeftModel.from_pretrained(m, checkpoint).merge_and_unload()
    return m.visual.eval().to("cuda")


def load_tower_repo(repo, dtype, trust_remote_code=False):
    """임의 HF 저장소 경로. `profile_other_vlm_families`의 견고한 로더를 재사용한다.

    거기서 이미 해결한 것들이 필요하다 — config 중첩 구조 보정, 비전 타워 없는
    모델 반환 방지, **가중치 미로드 검증**(키 규약이 다르면 from_pretrained가 경고만
    찍고 랜덤 초기화 모델을 돌려준다).
    """
    import transformers as HF
    from transformers import AutoConfig
    from profile_other_vlm_families import (find_vision_tower, check_weights_loaded)

    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=trust_remote_code)
        if isinstance(getattr(cfg, "text_config", None), dict):
            keys = ("hidden_size", "num_hidden_layers", "num_attention_heads")
            if all(getattr(cfg, k, None) == cfg.text_config.get(k)
                   for k in keys if k in cfg.text_config):
                delattr(cfg, "text_config")
    except Exception:
        pass
    order = [a for a in (getattr(cfg, "architectures", None) or [])
             if getattr(HF, a, None) is not None]
    order += ["AutoModelForVision2Seq", "AutoModelForImageTextToText", "AutoModel"]

    last = None
    for cn in order:
        cls = getattr(HF, cn, None)
        if cls is None:
            continue
        try:
            kw = dict(torch_dtype=getattr(torch, dtype),
                      trust_remote_code=trust_remote_code)
            if cfg is not None and cn != "AutoModel":
                kw["config"] = cfg
            m = cls.from_pretrained(repo, **kw)
            find_vision_tower(m)
            check_weights_loaded(m, repo, cn)
            print(f"  로더: {cn} → {type(m).__name__}")
            return find_vision_tower(m).eval().to("cuda")
        except Exception as e:
            last = e
    raise RuntimeError(f"로드 실패: {last}")


def tower_forward(tower, pv, grid):
    """계열마다 타워 시그니처가 다르다 — grid를 받는 쪽과 아닌 쪽.

    실측(2026-09-15): SmolVLM(Idefics3) 프로세서는 pixel_values를
    (B, num_tiles, C, H, W) 5-D로 내놓는데 `Idefics3VisionTransformer.forward`는
    4-D를 기대해 `too many values to unpack (expected 4)`로 죽는다.
    타일 축을 배치 축에 접어 넣는다 — 타워는 타일을 독립 이미지로 처리하므로
    활성 크기 측정에는 영향이 없다.
    """
    if grid is None and pv.dim() == 5:
        pv = pv.flatten(0, 1)
    try:
        out = tower(pv, grid) if grid is not None else tower(pv)
    except TypeError:
        out = tower(pv)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out


def get_blocks(tower):
    """계열마다 블록 리스트 이름이 다르다 — `blocks`(Qwen), `encoder.layers`(Idefics3/CLIP) 등.

    실측(2026-09-15): `tower.blocks`를 하드코딩했더니 SmolVLM(Idefics3)에서
    `'Idefics3VisionTransformer' object has no attribute 'blocks'`로 죽었다.
    `profile_other_vlm_families.find_blocks`가 이미 이름·타입으로 찾으므로 재사용한다.
    """
    if hasattr(tower, "blocks"):
        return tower.blocks
    from profile_other_vlm_families import find_blocks
    return find_blocks(tower)[1]


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
    blocks = get_blocks(tower)
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
        out = tower_forward(tower, pv, grid)
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
        worst = max(vals) if vals else float("nan")
        # **초과 비율** — 이미지별 max가 형식 상한을 넘는 비율.
        # 게이트 보정(2026-09-15)에서 이 값이 실제 붕괴율을 250장·6개 모델 기준
        # 평균 2.1%p(최대 6.4%p) 오차로 맞혔다. 임계값도 통계량 선택도 필요 없는
        # **직접 추정치**이므로 도구의 1차 출력으로 쓴다.
        exceed = (sum(1 for v in vals if v > fmax) + (len(rec[i]) - len(vals))) / \
                 max(len(rec[i]), 1)
        per_block.append({"block": i, "max_p50": p50, "max_worst": worst,
                          "headroom": (fmax / p50) if p50 and math.isfinite(p50) else None,
                          "headroom_worst": (fmax / worst) if worst and math.isfinite(worst) else None,
                          "exceed_rate": exceed,
                          "n_finite": len(vals), "n_images": len(rec[i])})
    return per_block, nan / len(cache)


def zero_obs_upper_bound(n, conf=0.95):
    """N장에서 초과가 0건일 때 참 초과율의 95% 상한 = 1 - (1-conf)^(1/N).

    한 번도 못 봤다고 0이 아니다. N=10이면 상한이 25.9%로, 넷 중 하나가 터지는
    모델도 '안전'으로 통과시킬 수 있다. 도구 기본값을 100장으로 둔 근거다.
    """
    return 1.0 - (1.0 - conf) ** (1.0 / max(n, 1))


def build_cache(processor, paths, dtype):
    """계열마다 grid_thw 유무가 다르므로 없으면 None을 넣는다."""
    cache = []
    for p in paths:
        enc = processor.image_processor(images=[Image.open(p).convert("RGB")],
                                        return_tensors="pt")
        pv = enc["pixel_values"]
        if pv.dim() > 2 and pv.shape[0] == 1:
            pv = pv[0] if pv.dim() == 3 else pv        # (1,N,D) → (N,D)
        grid = enc.get("image_grid_thw")
        cache.append((pv.to("cuda", getattr(torch, dtype)),
                      grid.to("cuda") if grid is not None else None))
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
    blk = get_blocks(tower)[block_idx]
    buf = {}

    def h_mlp(_m, _i, out):
        buf["mlp"] = (out[0] if isinstance(out, tuple) else out).detach()

    def h_blk(_m, _i, out):
        buf["out"] = (out[0] if isinstance(out, tuple) else out).detach()

    hs = [blk.mlp.register_forward_hook(h_mlp), blk.register_forward_hook(h_blk)]
    pairs = []
    for pv, grid in cache:
        buf.clear()
        tower_forward(tower, pv, grid)
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
        blk = get_blocks(tower)[item["block"]]
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
    ap.add_argument("--repo", default=None,
                    help="임의 HF 저장소를 대상으로 한다 (--checkpoint 대신)")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dtype", default="float16", choices=list(FORMAT_MAX))
    ap.add_argument("--max_pixels", type=int, default=None,
                    help="미지정이면 **모델 기본 설정**을 쓴다 — 기본 경로에서의 위험을 보려면 그대로 둘 것")
    ap.add_argument("--min_pixels", type=int, default=None)
    ap.add_argument("--image_dir", default=None, help="미지정이면 DriveLM val CAM_FRONT")
    ap.add_argument("--limit", type=int, default=100,
                    help="측정 이미지 수. 기본 100장 — 초과 0건일 때 참 초과율의 "
                         "95% 상한이 3.0%다(10장이면 25.9%로 너무 느슨하다)")
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
    src = args.repo or args.base
    try:
        proc = AutoProcessor.from_pretrained(
            src, trust_remote_code=args.trust_remote_code, **kw)
    except Exception:
        # max_pixels 같은 인자를 받지 않는 계열이 있다(SmolVLM 등). 로드 후 주입한다.
        proc = AutoProcessor.from_pretrained(
            src, trust_remote_code=args.trust_remote_code)
        for k, v in kw.items():
            if hasattr(proc.image_processor, k):
                setattr(proc.image_processor, k, v)
    ip = proc.image_processor
    if not hasattr(ip, "max_pixels"):
        # 고정 해상도 계열 — 해상도 인자가 의미 없다는 사실을 기록한다
        print("  (이 계열의 image_processor에는 max_pixels가 없다 — 고정/자체 규칙)")
        class _S: pass
        ip = _S(); ip.max_pixels = -1; ip.min_pixels = -1
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
    if args.repo:
        tower = load_tower_repo(args.repo, MAG, args.trust_remote_code)
    else:
        tower = load_tower(args.checkpoint, args.base, MAG)
    cache_mag = build_cache(proc, paths, MAG)
    if cache_mag[0][1] is None:
        print("  (grid_thw 없음 — 타워를 grid 없이 호출한다)")
    g0 = cache_mag[0][1]
    print(f"[형상] pixel_values={tuple(cache_mag[0][0].shape)}  "
          f"grid={g0.tolist() if g0 is not None else '없음'}")
    print(f"[측정] 크기={MAG} (참값)  붕괴={args.dtype}\n")

    before, _ = profile(tower, cache_mag, MAG, fmax_dtype=args.dtype)
    risky = plan_fix(before, args.target_headroom,
                     tower=tower, cache=cache_mag, dtype=args.dtype)

    # **가중치를 건드리기 전에 형식부터 검토하게 한다.**
    # 여유는 형식마다 다른 값이다 — 같은 활성이 fp16에서는 넘치고 bf16에서는 한참 남는다.
    # 가장 명백한 대안(bf16으로 빌드)을 도구가 먼저 제시하지 않으면 정직하지 않다.
    last_mag = before[-1]["max_p50"]
    alt = {dt: FORMAT_MAX[dt] / last_mag for dt in FORMAT_MAX if last_mag}
    print(f"\n[형식별 여유] 마지막 블록 max|act| = {last_mag:,.0f}")
    for dt, h in alt.items():
        mark = "  ← 목표 dtype" if dt == args.dtype else ""
        safe = "안전" if h >= args.target_headroom else "**위험**"
        print(f"  {dt:<10} {h:>12,.2f}배  {safe}{mark}")
    res_alt = {dt: h for dt, h in alt.items()}
    safer = [dt for dt, h in alt.items()
             if dt != args.dtype and h >= args.target_headroom]
    if risky and safer:
        print(f"\n  **권고: 가중치를 고치기 전에 {' 또는 '.join(safer)} 빌드를 먼저 검토할 것.**")
        print(f"  {args.dtype}에서 위험한 것이지 모델이 잘못된 것이 아니다. 형식을 바꾸면")
        print(f"  교정 없이 해결되며, 그 대가(지연시간·정확도)는 별도로 재야 한다.")

    # ---- 판정 (2026-09-15 게이트 보정 결과 반영) ----
    # 근거: eval_results/gate_calibration.json, detection_curve{,_fine}.json
    #  - 전이는 **최악 여유 1.00~1.05**에서 일어난다(물리적 한계 1.0과 일치).
    #    기존 임계값 2.0은 그 경계보다 1.91배 보수적이고, 20개 합성 점에서
    #    오탐 7건·미탐 0건을 냈다.
    #  - p50/p95/최악 세 통계량은 여백 비율이 1.043/1.043/1.045로 **사실상 동등**하다.
    #    한 모델 안에서는 셋이 비례해 움직이기 때문이다. 최악을 쓰는 이유는 여백이
    #    커서가 아니라 **임계값이 물리적 의미(형식 상한)를 갖기 때문**이다.
    #  - 그래서 1차 판정은 임계값이 아니라 **초과 비율**로 한다.
    lastb = before[-1]
    exc, nimg = lastb["exceed_rate"], lastb.get("n_images", len(cache_mag))
    ub = zero_obs_upper_bound(nimg)
    hw = lastb.get("headroom_worst")
    print(f"\n[판정] 마지막 블록 (기준: 초과 비율, {nimg}장)")
    print(f"  초과 비율 {exc*100:.1f}%  ← 붕괴율 추정치 "
          f"(실측 대조 평균 오차 2.1%p, 6개 모델)")
    print(f"  여유  p50 {lastb['headroom']:.2f}배 · 최악 {hw:.2f}배" if hw else "")
    if exc > 0:
        print(f"  → **위험.** {nimg}장 중 {round(exc*nimg)}장이 이미 형식 상한을 넘는다.")
    elif hw is not None and hw < MARGINAL_HEADROOM:
        print(f"  → **경계.** 초과는 없으나 최악 여유가 {hw:.2f}배로 "
              f"{MARGINAL_HEADROOM}배 미만이다. 표본 밖 이미지가 넘을 수 있다.")
    else:
        print(f"  → 안전. 다만 {nimg}장에서 0건이 관측됐을 뿐이므로 "
              f"참 초과율의 95% 상한은 **{ub*100:.1f}%**다.")
    if nimg < 100:
        print(f"  ⚠ 표본 {nimg}장은 적다 — 상한 {ub*100:.1f}%. "
              f"100장(3.0%) 이상을 권한다.")

    # 붕괴는 목표 dtype으로 갈아끼워 잰다
    def collapse_in_target():
        tower.to(getattr(torch, args.dtype))
        c = build_cache(proc, paths, args.dtype)
        n = 0
        with torch.no_grad():
            for pv, grid in c:
                if not torch.isfinite(tower_forward(tower, pv, grid)).all():
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

    res = {"target": args.repo or args.checkpoint, "dtype": args.dtype,
           "max_pixels": ip.max_pixels, "n_images": len(paths),
           "target_headroom": args.target_headroom,
           "before": {"per_block": before, "nan_rate": nan_before},
           "headroom_by_format": res_alt,
           "safer_formats": safer,
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
                      "(배포 조건 실측: 원본 해상도·fp16·1,029문항 짝지음에서 "
                      "무교정 bf16 대비 +0.78%p, McNemar p=0.34 — 측정 한계 이하. "
                      "다만 비유의는 동등성의 증거가 아니다).")

            if args.save_weights:
                torch.save(tower.state_dict(), args.save_weights)
                print(f"\n  교정된 비전 타워 저장: {args.save_weights}")
                print("  **적재 절차** — 이 state_dict는 LoRA를 **병합한** 타워 기준이다.")
                print("    (1) 적용하는 쪽에서 먼저 merge_and_unload()")
                print("    (2) 텍스트 모델이 아니라 비전 타워(.visual)에 얹을 것")
                print("    (3) 빠진 파라미터가 하나라도 있으면 **예외로 중단**할 것 —")
                print("        strict 검증 없이 적재하면 조용히 무교정 모델을 평가하게 된다")

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
