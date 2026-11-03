"""
실험 5 — 고정 배수 대조. 2026-11-03.
사전 등록: thesis_outline_20260910.md §6 「교정의 전 계열 확장」 개정 8 (결과 보기 전).

기존 처방(Draw Things: MLP 하향 사상에 ⅛·¼ 고정 축소)과 도구의 교정을 같은 모델에서 나란히 잰다.
  F¼·F⅛ — 측정 없이 **모든 블록**의 down_proj 입력을 s배(가중치 ×s, 편향 그대로) — 비등가 고정 처방
  P⅛    — 도구의 곱셈 이전 축소를 계산값 f = 0.3554 대신 ⅛로(마지막 블록) — 비등가 과잉 축소
  R⅛    — 잔차 재조정을 c = ¼ 대신 ⅛로(eps ×c²) — 등가 과잉 축소(fp32 등가성 검사 통과가 적용 조건)

Qwen2.5-VL: 팔마다 fp16 상실률, fp16 타워 출력의 fp32(무교정) 대비 상대 오차를 799장으로 재고, 정확도 평가용 가중치를 저장한다
(`<state_dir>/baseline_v2_exp5_<팔>.pt` + `.fix.json`). Gemma 4: F¼·F⅛의 승격 없는 경로 (ii) 상실률.
팔 단위로 이어 한다(결과 파일에 끝난 팔이 있으면 건너뛴다).

    python scripts/fixed_mult_arms.py qwen --checkpoint checkpoints/student_baseline_v2/epoch_1 \
        --max_pixels 1440000 --min_pixels 3136 --out eval_results/fixfam2/exp5_qwen_arms.json
    python scripts/fixed_mult_arms.py gemma --model_dir /workspace/gemma4_vision/gemma-4-26B-A4B-it --max_soft_tokens 280 \
        --out eval_results/fixfam2/exp5_gemma_26B_st280.json
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                       # noqa: E402

TARGET = 2.0


def _save(out, res):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out + ".tmp").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    Path(out + ".tmp").replace(out)


def _load(out):
    return json.loads(Path(out).read_text()) if Path(out).exists() else {}


@torch.no_grad()
def scale_down_proj(tower, blocks_of, mlp_of, s):
    """F — 모든 블록의 down_proj 가중치 ×s(편향 그대로) = down_proj 입력을 s배 한 것과 같다."""
    n = 0
    for blk in blocks_of(tower):
        dp = getattr(mlp_of(blk), "down_proj", None)
        # Gemma 4는 down_proj가 감싸개(Gemma4ClippableLinear: 입력·출력 자르기 + 안쪽 .linear)다 — 안쪽 선형층을 줄인다
        lin = dp if isinstance(dp, torch.nn.Linear) else getattr(dp, "linear", None)
        if not isinstance(lin, torch.nn.Linear):
            raise RuntimeError("down_proj(선형층)가 없는 블록 — 고정 처방을 적용할 수 없다")
        lin.weight.mul_(s)
        n += 1
    return n


def run_qwen(a):
    import headroom_guard as G
    from transformers import AutoProcessor
    from dataloader import build_token_to_images
    res = _load(a.out)
    res.setdefault("arms", {})
    res.update({"checkpoint": a.checkpoint, "max_pixels": a.max_pixels, "n_images": a.limit})
    proc = AutoProcessor.from_pretrained(G.BASE_3B, max_pixels=a.max_pixels, min_pixels=a.min_pixels)
    tok = build_token_to_images(G.DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]
    tower0 = G.load_tower(a.checkpoint, G.BASE_3B, "float32").cpu()     # 교정 전(fp32, CPU 사본)
    cache32 = G.build_cache(proc, paths, "float32", device="cpu")
    fwd32 = lambda tw, x: G.tower_forward(tw, x[0].to("cuda"), x[1].to("cuda") if x[1] is not None else None)
    fwd16 = lambda tw, x: G.tower_forward(tw, x[0].to("cuda", torch.float16), x[1].to("cuda") if x[1] is not None else None)

    # 기준 출력 — 무교정 fp32 (CPU에 둔다)
    ref = None
    for arm in a.arms.split(","):
        if arm in res["arms"] and Path(f"{a.state_dir}/baseline_v2_exp5_{arm}.pt").exists():
            print(f"[{arm}] 끝난 팔 — 건너뛴다"); continue
        if ref is None:
            r32 = copy.deepcopy(tower0).cuda()
            with torch.no_grad():
                ref = [fwd32(r32, x).float().cpu() for x in cache32]
            del r32; torch.cuda.empty_cache()
        t = copy.deepcopy(tower0).cuda()           # fp32에서 편집하고, 측정은 fp16 사본으로
        info, side, comp = {"arm": arm}, {"version": 1}, 1.0
        blocks = G.get_blocks(t)
        if arm in ("F4", "F8"):
            s = 0.25 if arm == "F4" else 0.125
            info["n_blocks"] = scale_down_proj(t, G.get_blocks, G._mlp_of, s)
            info["s"] = s
        elif arm == "P8":
            applied = G.apply_fix(t, [{"block": len(blocks) - 1, "scale": 8.0, "headroom": None}])
            info["applied"] = applied
        elif arm == "R8":
            c = 0.125
            plan = HX.plan_residual(t, G.get_blocks, cache32[:8], fwd32, c)
            if not plan["ok"]:
                info["failed"] = plan["reason"]; res["arms"][arm] = info; _save(a.out, res); continue
            t_c = HX.apply_residual(t, G.get_blocks, plan["entry"], c)
            eps = HX.apply_eps(t_c, plan["eps_norms"], c)
            eq = HX.compare(t, t_c, cache32, fwd32, comp=plan["comp"])
            info.update({"c": c, "entry": plan["entry"], "exit": plan["exit"], "equivalence": eq,
                         "accepted": eq["cos_min"] >= 0.99999})
            print(f"[R8] 등가성 코사인 최소 {eq['cos_min']:.8f}")
            comp = plan["comp"]
            del t, plan                                # 계획이 교정 전 fp32 타워를 붙잡고 있다 — fp16 측정 전에 놓는다
            torch.cuda.empty_cache()
            t = t_c; side["eps"] = eps
        else:
            raise SystemExit(f"모르는 팔: {arm}")
        # fp16 상실률과 상대 오차
        t16 = copy.deepcopy(t).half()
        bad, errs = 0, []
        with torch.no_grad():
            for x, y in zip(cache32, ref):
                z = fwd16(t16, x).float().cpu() * comp
                if not torch.isfinite(z).all():
                    bad += 1; errs.append(float("inf")); continue
                errs.append(float((z.double() - y.double()).norm() / y.double().norm()))
        fe = sorted(e for e in errs if math.isfinite(e))
        info.update({"loss_fp16": bad / len(cache32),
                     "fp16_error": {"median": fe[len(fe) // 2] if fe else None,
                                    "p99": fe[min(len(fe) - 1, int(0.99 * len(fe)))] if fe else None,
                                    "n_finite": len(fe), "n": len(errs)}})
        print(f"[{arm}] fp16 상실 {bad}/{len(cache32)} · 상대 오차 중앙 {info['fp16_error']['median']} · p99 {info['fp16_error']['p99']}")
        # 정확도 평가용 저장 (병합된 타워 기준 state_dict — eval_nuscenesqa_val --vision_state)
        Path(a.state_dir).mkdir(parents=True, exist_ok=True)
        sp = f"{a.state_dir}/baseline_v2_exp5_{arm}.pt"
        torch.save({k: v.detach().cpu() for k, v in t.state_dict().items()}, sp + ".tmp")
        Path(sp + ".tmp").replace(sp)
        Path(sp + ".fix.json").write_text(json.dumps(side, indent=1))
        info["state"] = sp
        res["arms"][arm] = info
        _save(a.out, res)
        del t, t16; torch.cuda.empty_cache()
    print(f"저장: {a.out}")


def run_gemma(a):
    import run_fix_gemma as R
    from probe_gemma4_vision import load_vision, build_inputs
    res = _load(a.out)
    res.setdefault("arms", {})
    res.update({"model_dir": a.model_dir, "max_soft_tokens": a.max_soft_tokens})
    paths, cache = build_inputs(a.model_dir, a.max_soft_tokens, a.limit)
    res["n_images"] = len(paths)
    from headroom_guard import _mlp_of
    if "base" not in res:
        vm16 = load_vision(a.model_dir, "float16")
        res["base"] = {"loss_no_upcast": R.loss_no_upcast(vm16, cache)}
        _save(a.out, res); del vm16; torch.cuda.empty_cache()
    for arm in a.arms.split(","):
        if arm in res["arms"]:
            print(f"[{arm}] 끝난 팔 — 건너뛴다"); continue
        s = {"F4": 0.25, "F8": 0.125}[arm]
        vm16 = load_vision(a.model_dir, "float16")
        n = scale_down_proj(vm16, R.blocks_of, _mlp_of, s)
        res["arms"][arm] = {"s": s, "n_blocks": n, "loss_no_upcast": R.loss_no_upcast(vm16, cache)}
        print(f"[{arm}] 경로 (ii) 상실 {res['base']['loss_no_upcast']:.1%} → {res['arms'][arm]['loss_no_upcast']:.1%}")
        _save(a.out, res); del vm16; torch.cuda.empty_cache()
    print(f"저장: {a.out}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    sub = ap.add_subparsers(dest="model", required=True)
    q = sub.add_parser("qwen")
    q.add_argument("--checkpoint", default="checkpoints/student_baseline_v2/epoch_1")
    q.add_argument("--max_pixels", type=int, default=1440000)
    q.add_argument("--min_pixels", type=int, default=3136)
    q.add_argument("--limit", type=int, default=799)
    q.add_argument("--arms", default="F4,F8,P8,R8")
    q.add_argument("--state_dir", default="checkpoints_fix")
    q.add_argument("--out", required=True)
    g = sub.add_parser("gemma")
    g.add_argument("--model_dir", required=True)
    g.add_argument("--max_soft_tokens", type=int, default=280)
    g.add_argument("--limit", type=int, default=799)
    g.add_argument("--arms", default="F4,F8")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    import awq_compat
    awq_compat.patch()
    (run_qwen if a.model == "qwen" else run_gemma)(a)


if __name__ == "__main__":
    main()
