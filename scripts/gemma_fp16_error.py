"""
Gemma 4 (R)의 fp16 출력 오차 — 실험 4와 같은 정의. 2026-11-04.
사전 등록: thesis_outline_20260910.md §6 「교정의 전 계열 확장」 개정 12 (결과 보기 전).

기준 y = 무교정 fp32 기준 구현(블록 뒤 연산 승격) 출력. 두 fp16 출력 z의 상대 오차 ‖z − y‖/‖y‖를 799장으로 잰다.
  (R) (ii) — run_fix_gemma.py와 같은 절차로 잔차 재조정(c, 입구, eps ×c²)을 적용한 fp16, 승격 없는 경로 (배포 경로). 출구 보정 comp를 곱한다.
  기준 (i) — 무교정 fp16, 블록 뒤 승격 (넘침을 fp32 승격으로 피한 fp16의 오차 — 기준선)

    python scripts/gemma_fp16_error.py --model_dir /workspace/gemma4_vision/gemma-4-26B-A4B-it --max_soft_tokens 280 \
        --out eval_results/fixfam2/exp5_gemma_err_26B-A4B_st280.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                                       # noqa: E402
from probe_gemma4_vision import load_vision, build_inputs, forward_points, FP16_MAX   # noqa: E402
from run_fix_gemma import blocks_of, fwd, TARGET                # noqa: E402


def stats(errs):
    fe = sorted(e for e in errs if math.isfinite(e))
    q = lambda p: fe[min(len(fe) - 1, int(p * len(fe)))] if fe else None
    return {"median": q(0.5), "p99": q(0.99), "max": fe[-1] if fe else None, "n_finite": len(fe), "n": len(errs)}


@torch.no_grad()
def rel_errors(vm, cache, ref, upcast, comp=1.0):
    errs = []
    for x, y in zip(cache, ref):
        z = forward_points(vm, x[0], x[1], upcast=upcast)["final"].float().cpu() * comp
        errs.append(float((z.double() - y.double()).norm() / y.double().norm()) if torch.isfinite(z).all() else float("inf"))
    return errs


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_soft_tokens", type=int, default=280)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths, cache = build_inputs(a.model_dir, a.max_soft_tokens, a.limit)
    res = {"model_dir": a.model_dir, "max_soft_tokens": a.max_soft_tokens, "n_images": len(paths)}

    # c — run_fix_gemma와 같이 bf16 연산 단위 진단의 최대에서
    vmb = load_vision(a.model_dir, "bfloat16")
    ops = HX.profile_ops(vmb, cache, fwd, FP16_MAX, TARGET)
    c = HX.choose_c(ops["global_max"]["max_worst"], FP16_MAX, TARGET)
    res["c"] = c
    del vmb; torch.cuda.empty_cache()

    # 기준 — 무교정 fp32 기준 구현
    vm32 = load_vision(a.model_dir, "float32")
    with torch.no_grad():
        ref = [fwd(vm32, x).float().cpu() for x in cache]
    plan = HX.plan_residual(vm32, blocks_of, cache[:8], fwd, c)
    if not plan["ok"]:
        raise SystemExit(f"잔차 재조정 불가 — {plan['reason']}")
    res["plan"] = {"entry": plan["entry"], "exit": plan["exit"], "comp": plan["comp"]}
    plan.pop("entry_module", None)              # 교정 전 fp32 타워를 붙잡고 있다 — 놓는다
    del vm32; torch.cuda.empty_cache()

    # 기준 (i) — 무교정 fp16, 승격
    vm16 = load_vision(a.model_dir, "float16")
    e_i = rel_errors(vm16, cache, ref, upcast=True)
    res["ref_path_i"] = stats(e_i)
    print(f"[기준 (i)] 무교정 fp16·승격 — 상대 오차 {res['ref_path_i']}")

    # (R) (ii) — 교정 fp16, 승격 없음
    HX.apply_residual(vm16, blocks_of, plan["entry"], c, inplace=True)
    HX.apply_eps(vm16, plan["eps_norms"], c)
    e_r = rel_errors(vm16, cache, ref, upcast=False, comp=plan["comp"])
    res["resid_path_ii"] = stats(e_r)
    print(f"[(R) (ii)] 교정 fp16·승격 없음 — 상대 오차 {res['resid_path_ii']}")
    m_i, m_r = res["ref_path_i"]["median"], res["resid_path_ii"]["median"]
    res["median_ratio_ii_over_i"] = (m_r / m_i) if (m_i and m_r) else None
    print(f"[비] (ii)/(i) 중앙값 {res['median_ratio_ii_over_i']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
