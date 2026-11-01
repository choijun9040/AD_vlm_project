"""
활성 상수 재매개화 · 포화 절단 · 키 중심화 — 4장 시험. 2026-11-01.
사전 등록: thesis_outline_20260910.md §6 「교정의 전 계열 확장」 개정 5 (결과 보기 전). 대기열에는 넣지 않는다.

도구의 등가 교정((R) 잔차 재조정, (A) 어텐션 스케일 접기, (G) 게이트 곱)으로도 남는 두 종류를 그래프 수정으로 다룰 수 있는지 본다.
- 팔 1 `act`  : CLIP QuickGELU `x·σ(1.702x)` — fc1 ×s, 상수 1.702/s, fc2 가중치 ×1/s (등가)
- 팔 2 `clamp`: σ 인자를 만들기 전에 x를 ±16/1.702로 자른다 (σ(±16)은 fp16에서 정확히 1·0 — fp32에서 오차 ≤ e⁻¹⁶)
- 팔 3 `key`  : InternViT 어텐션 — 키 평균(앞 절반 이미지로 추정)을 qkv 편향의 k 부분에서 뺀다 (q·b는 행마다 상수 → 소프트맥스 불변)
세 팔 모두 (R)을 먼저 적용한 타워 위에 얹고, (R)만 적용한 fp32 타워와 출력을 비교한다(등가성 ②).

    python scripts/probe_act_key_reparam.py --repo Isotr0py/Phi-3.5-vision-instruct-AWQ --trust_remote_code --arms act,clamp \
        --out logs/probeAK/phi35.json
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_guard as G                      # noqa: E402
import headroom_ext as HX                       # noqa: E402
from dataloader import build_token_to_images   # noqa: E402

TARGET = 2.0
CLAMP_K = 16.0


class QuickGELUScaled(nn.Module):
    """x·σ(a·x) — 팔 1. a = 1.702/s 이면 입력이 s배가 된 뒤에도 출력이 정확히 s배다."""
    def __init__(self, a):
        super().__init__()
        self.a = a

    def forward(self, x):
        return x * torch.sigmoid(self.a * x)


class QuickGELUClamped(nn.Module):
    """x·σ(1.702·clip(x)) — 팔 2. σ 인자가 ±k를 넘지 않게 x를 먼저 자른다(곱하는 x는 자르지 않는다)."""
    def __init__(self, k=CLAMP_K):
        super().__init__()
        self.k = k

    def forward(self, x):
        z = x.clamp(-self.k / 1.702, self.k / 1.702)
        return x * torch.sigmoid(1.702 * z)


def block_rows(ops, pre, i, sub):
    return [r for r in ops["rows"] if r["point"].startswith(f"{pre}{i}.{sub}")]


def worst(ops, fm):
    g = ops["global_max"]
    return {"point": g["point"], "max": g["max_worst"], "headroom": fm / g["max_worst"]}


@torch.no_grad()
def arm_act(tower, ops, fm, kind):
    """위험 블록(활성 쪽 연산이 목표를 넘는 블록)에 팔 1 또는 팔 2."""
    blocks = G.get_blocks(tower)
    pre = HX._block_prefix(tower, blocks)
    tmax, done = fm / TARGET, []
    for i, b in enumerate(blocks):
        mlp = G._mlp_of(b)
        act = getattr(mlp, "activation_fn", None)
        if act is None or "QuickGELU" not in type(act).__name__ or not hasattr(mlp, "fc1") or not hasattr(mlp, "fc2"):
            continue
        rows = block_rows(ops, pre, i, "mlp.")
        mx = max((r["max_worst"] for r in rows), default=0.0)
        if mx <= tmax:
            continue
        if kind == "act":
            s = HX.choose_c(mx, fm, TARGET)
            for t in HX._affine(mlp.fc1):
                t.mul_(s)
            mlp.fc2.weight.div_(s)
            mlp.activation_fn = QuickGELUScaled(1.702 / s)
            done.append({"block": i, "s": s, "mlp_max": mx})
        else:
            mlp.activation_fn = QuickGELUClamped()
            done.append({"block": i, "k": CLAMP_K, "mlp_max": mx})
    return done


@torch.no_grad()
def arm_key(tower, est_cache, fwd):
    """팔 3 — 블록마다 키 평균을 qkv 편향의 k 부분에서 뺀다."""
    blocks = G.get_blocks(tower)
    sums, cnt, hs = {}, {}, []
    for i, b in enumerate(blocks):
        at = getattr(b, "attn", None)
        if at is None or not isinstance(getattr(at, "qkv", None), nn.Linear) or getattr(at, "qk_normalization", False):
            continue

        def h(_m, _a, o, i=i):
            C = o.shape[-1] // 3
            k = o[..., C:2 * C].detach().double().reshape(-1, C)
            sums[i] = sums.get(i, 0) + k.sum(0)
            cnt[i] = cnt.get(i, 0) + k.shape[0]
        hs.append(at.qkv.register_forward_hook(h))
    for x in est_cache:
        fwd(tower, x)
    for hk in hs:
        hk.remove()
    done = []
    for i in sums:
        qkv = blocks[i].attn.qkv
        C = qkv.out_features // 3
        mean = (sums[i] / cnt[i]).to(qkv.weight.dtype)
        if qkv.bias is None:
            qkv.bias = nn.Parameter(torch.zeros(qkv.out_features, dtype=qkv.weight.dtype, device=qkv.weight.device))
        qkv.bias[C:2 * C] -= mean
        done.append({"block": i, "mean_norm": float(mean.float().norm())})
    return done


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--arms", default="act,clamp")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tower, cfg = G.load_tower_direct(a.repo, "bfloat16", a.trust_remote_code)
    ip = G.load_image_processor(a.repo, cfg, a.trust_remote_code)
    tok = build_token_to_images(G.DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]
    cache = G.build_cache(ip, paths, "bfloat16")
    cache32 = G.build_cache(ip, paths, "float32", device="cpu")
    fwd = lambda tw, x: G.tower_forward(tw, x[0], x[1])
    fwd32 = lambda tw, x: G.tower_forward(tw, x[0].to("cuda"), x[1].to("cuda") if x[1] is not None else None)
    with torch.no_grad():
        fwd(tower, cache[0])                      # 예열 (headroom_guard와 같다)

    ops0 = HX.profile_ops(tower, cache, fwd, 65504.0, TARGET)
    fm = HX.virtual_fmax(ops0, 1.5)
    ops0 = HX.profile_ops(tower, cache, fwd, fm, TARGET)
    res = {"repo": a.repo, "n_images": len(paths), "virtual_fmax": fm, "before": worst(ops0, fm)}

    # (R) — 도구와 같은 절차
    c = HX.choose_c(ops0["global_max"]["max_worst"], fm, TARGET)
    t32 = copy.deepcopy(tower).float()
    plan = HX.plan_residual(t32, G.get_blocks, cache32[:min(8, len(cache32))], fwd32, c)
    if not plan["ok"]:
        res["residual"] = {"ok": False, "reason": plan["reason"]}
        print(f"[R] 불가 — {plan['reason']}")
    else:
        HX.apply_residual(t32, G.get_blocks, plan["entry"], c, inplace=True)
        HX.apply_eps(t32, plan["eps_norms"], c)
        HX.apply_residual(tower, G.get_blocks, plan["entry"], c, inplace=True)
        HX.apply_eps(tower, plan["eps_norms"], c)
        res["residual"] = {"ok": True, "c": c, "entry": plan["entry"]}
    opsR = HX.profile_ops(tower, cache, fwd, fm, TARGET)
    res["after_R"] = worst(opsR, fm)
    print(f"[R] 최대 {res['after_R']['max']:,.1f} ({res['after_R']['point']}) · 최악 여유 {res['after_R']['headroom']:.2f}배")

    for arm in a.arms.split(","):
        tw, tw32 = copy.deepcopy(tower), copy.deepcopy(t32)
        if arm in ("act", "clamp"):
            done = arm_act(tw, opsR, fm, arm)
            arm_act(tw32, opsR, fm, arm)
        elif arm == "key":
            half = cache[:max(1, len(cache) // 2)]
            done = arm_key(tw, half, fwd)
            arm_key(tw32, [(x[0].float(), x[1]) for x in half], fwd)    # 같은 평균을 fp32에서 다시 잰다
        else:
            raise SystemExit(f"모르는 팔: {arm}")
        eq = HX.compare(t32, tw32, cache32, fwd32)
        ops_a = HX.profile_ops(tw, cache, fwd, fm, TARGET)
        w = worst(ops_a, fm)
        res[arm] = {"blocks": done, "equivalence": eq, "after": w,
                    "top": [(r["point"], r["max_worst"], fm / r["max_worst"]) for r in
                            sorted(ops_a["rows"], key=lambda r: -r["max_worst"])[:4]]}
        print(f"[{arm}] 블록 {len(done)}개 · 등가성 코사인 최소 {eq['cos_min']:.8f} · 크기 비율 "
              f"{eq['ratio_min']:.6f}~{eq['ratio_max']:.6f} · 최대 {w['max']:,.1f} ({w['point']}) · 최악 여유 {w['headroom']:.2f}배")
        for p, m, h in res[arm]["top"]:
            print(f"      {p}  {m:,.1f}  {h:.2f}배")
        del tw, tw32
        torch.cuda.empty_cache()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
