"""
Gemma 4 비전 인코더 — 연산 단위 진단과 잔차 재조정의 도구 경로 검증. 2026-10-30.
사전 등록: thesis_outline_20260910.md §6 「교정의 전 계열 확장」(2026-10-30 23:20 KST, 결과 보기 전).

`probe_gemma4_vision.py`는 지점을 손으로 골라 쟀다. 여기서는 **도구의 범용 경로**(headroom_ext)가 같은 결과를 내는지 본다 —
지점을 알려 주지 않고 연산 단위 진단이 pooler의 × √d를 찾는지, 입구·출구를 스스로 찾아 등가 교정을 하는지.

판정 기준(사전 등록): ① 처음 넘는 연산이 pooler 스케일 ② fp32 등가성 코사인 최소 ≥ 0.99999 ③ 경로 (ii) 상실 0/799.

    python scripts/run_fix_gemma.py --model_dir /workspace/gemma4_vision/gemma-4-26B-A4B-it --max_soft_tokens 280 \
        --limit 799 --out eval_results/fixfam/gemma4_26B_st280.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                                       # noqa: E402
from probe_gemma4_vision import load_vision, build_inputs, forward_points, FP16_MAX   # noqa: E402

TARGET = 2.0


def blocks_of(vm):
    return vm.encoder.layers


def fwd(vm, x):
    return forward_points(vm, x[0], x[1], upcast=True)["final"]      # 기준 구현(승격 있음)


@torch.no_grad()
def loss_no_upcast(vm, cache):
    """경로 (ii) — 블록 뒤 두 연산까지 fp16(수정 전 vLLM·fp16 엔진 모사). 비유한 출력 비율."""
    bad = 0
    for pv, pos in cache:
        out = forward_points(vm, pv, pos, upcast=False)["final"]
        bad += int(not torch.isfinite(out).all())
    return bad / len(cache)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_soft_tokens", type=int, default=280)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths, cache = build_inputs(args.model_dir, args.max_soft_tokens, args.limit)
    res = {"model_dir": args.model_dir, "max_soft_tokens": args.max_soft_tokens, "n_images": len(paths)}

    vm = load_vision(args.model_dir, "bfloat16")                    # 크기는 bf16에서 (측정 규칙)
    ops = HX.profile_ops(vm, cache, fwd, FP16_MAX, TARGET)
    res["ops_before"] = HX.summarize_ops(ops)
    fe, fa = ops["first_exceed"], ops["first_exceed_no_upcast"]
    print(f"[진단] 모델 형식 처음 초과: {fe['point'] if fe else '없음'} · 승격 없는 경로 처음 초과: "
          f"{fa['point'] if fa else '없음'} · 최대 {ops['global_max']['max_worst']:,.0f} ({ops['global_max']['point']})")
    c = HX.choose_c(ops["global_max"]["max_worst"], FP16_MAX, TARGET)
    res["c"] = c

    vm16 = load_vision(args.model_dir, "float16")
    res["loss_no_upcast_before"] = loss_no_upcast(vm16, cache)
    print(f"[상실] 경로 (ii) 교정 전 {res['loss_no_upcast_before']:.1%}")

    vm32 = load_vision(args.model_dir, "float32")
    plan = HX.plan_residual(vm32, blocks_of, cache[:8], fwd, c)
    res["plan"] = {k: v for k, v in plan.items() if k != "entry_module"}
    if not plan["ok"]:
        print(f"[교정] 잔차 재조정 불가 — {plan['reason']}")
    else:
        vm32c = HX.apply_residual(vm32, blocks_of, plan["entry"], c)
        eq = HX.compare(vm32, vm32c, cache, fwd, comp=plan["comp"])
        res["equivalence_fp32"] = eq
        del vm32c
        print(f"[교정] c={c:g} 입구 {plan['entry']} 출구 {plan['exit']} 쓰기 {plan['writers']} · "
              f"등가성 코사인 최소 {eq['cos_min']:.8f}")
        HX.apply_residual(vm16, blocks_of, plan["entry"], c, inplace=True)
        res["loss_no_upcast_after"] = loss_no_upcast(vm16, cache)
        HX.apply_residual(vm, blocks_of, plan["entry"], c, inplace=True)
        ops_a = HX.profile_ops(vm, cache, fwd, FP16_MAX, TARGET)
        res["ops_after"] = HX.summarize_ops(ops_a)
        print(f"[검증] 경로 (ii) 상실 {res['loss_no_upcast_before']:.1%} → {res['loss_no_upcast_after']:.1%} · "
              f"교정 후 최대 {ops_a['global_max']['max_worst']:,.0f} (여유 {FP16_MAX/ops_a['global_max']['max_worst']:.2f}배)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
