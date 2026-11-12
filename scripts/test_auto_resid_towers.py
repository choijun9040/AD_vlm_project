"""(R) 자동 도구 시험 — Qwen2.5-VL 기준 모델 비전 타워 / Gemma 4 비전 인코더 (개정 18 ③).
진단(bf16, 원본 해상도 8장) → 목표 초과 연산 전부 → 작은 입력으로 그래프 기록 → 영역 → c → fp32 등가성(8장) → fp16 비유한. 1차 교정 실험의 구성(plan_residual)과 비교한다.
    venv_main:  python scripts/test_auto_resid_towers.py qwen
    venv_gemma: python scripts/test_auto_resid_towers.py gemma
개정 36 (2026-11-09): --calib data/drivelm_official/train100.jsonl --out <json> — 진단 입력을 c와 같은 측정용 100장으로, Qwen은 공식 처리기 기본값,
등가성·fp16 비유한도 그 100장에서. 옵션이 없으면 예전 시험 그대로(검증 앞 8장).
"""
import copy, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_resid as AR
import headroom_ext as HX
import argparse
_ap = argparse.ArgumentParser(allow_abbrev=False)
_ap.add_argument("which"); _ap.add_argument("ckpt", nargs="?", default="none")
_ap.add_argument("--calib", default=None); _ap.add_argument("--out", default=None)
_a = _ap.parse_args()
which, CKPT = _a.which, _a.ckpt                               # 공식 저장소 모델(none) — 2026-11-04 사용자 결정
N = 8
CAL = [json.loads(l)["image"] for l in open(_a.calib)] if _a.calib else None
RES = {"which": which, "calib": _a.calib, "n_diag": len(CAL) if CAL else N}
if which == "qwen":
    import awq_compat; awq_compat.patch()
    import headroom_guard as G
    from transformers import AutoProcessor
    from dataloader import build_token_to_images
    tok = build_token_to_images(G.DRIVELM_VAL)
    paths = CAL or [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:N]
    t32 = G.load_tower(CKPT, G.BASE_3B, "float32")
    print("체크포인트", CKPT, "(none = 공식 Qwen/Qwen2.5-VL-3B-Instruct)")
    proc_big = AutoProcessor.from_pretrained(G.BASE_3B) if CAL else AutoProcessor.from_pretrained(G.BASE_3B, max_pixels=1440000, min_pixels=3136)
    big = G.build_cache(proc_big, paths, "float32", device="cpu")
    small = G.build_cache(AutoProcessor.from_pretrained(G.BASE_3B, max_pixels=50176, min_pixels=3136), paths[:1], "float32", device="cpu")[0]
    fwd = lambda tw, x: G.tower_forward(tw, x[0].to("cuda", next(tw.parameters()).dtype), x[1].to("cuda") if x[1] is not None else None)
    model = t32
else:
    from probe_gemma4_vision import load_vision, build_inputs, forward_points
    md = "/workspace/gemma4_vision/gemma-4-26B-A4B-it"
    if CAL:
        from transformers import AutoImageProcessor
        from PIL import Image
        _ip = AutoImageProcessor.from_pretrained(md, max_soft_tokens=280)
        big = [(e["pixel_values"], e["image_position_ids"]) for e in (_ip(images=[Image.open(q).convert("RGB")], return_tensors="pt") for q in CAL)]
    else:
        _p, big = build_inputs(md, 280, N)
    _p, sm = build_inputs(md, 70, 1); small = sm[0]
    model = load_vision(md, "float32")
    fwd = lambda m, x: forward_points(m, x[0], x[1], upcast=False)["final"]
mb = copy.deepcopy(model).to(torch.bfloat16)
ops = HX.profile_ops(mb, big, fwd, 65504.0, 2.0)
over = [r for r in ops["rows"] if r["max_worst"] > 65504.0 / 2 and r["max_worst"] < 1e30]
mx = max(r["max_worst"] for r in over)
print("목표 초과 연산", len(over), "최댓값", mx, [r["point"] for r in over][:10])
RES.update({"n_over_target": len(over), "max_worst": mx, "point": max(over, key=lambda r: r["max_worst"])["point"]})
del mb; torch.cuda.empty_cache()
Gr = AR.trace(model, lambda m: fwd(m, small))
tg, miss = AR.targets_from_points(Gr, [r["point"] for r in over])
print("그래프 연산", len(Gr["nodes"]), "출발점", len(tg), "못 찾음", miss[:5])
RES.update({"graph_ops": len(Gr["nodes"]), "targets": len(tg), "missing": miss})
plan = AR.solve(Gr, targets=tg)
show = {k: (v if not isinstance(v, list) or len(v) < 12 else v[:4] + [f"… 총 {len(v)}개"]) for k, v in plan.items() if k not in ("scale", "eps")}
print(json.dumps(show, ensure_ascii=False, indent=1))
if plan.get("ok"):
    eps = sorted(plan["eps"]); print("eps 모듈", len(eps), eps[:3], "…", eps[-2:])
    print("배수 매개변수", len(plan["scale"]), "×1/c:", [k for k, v in plan["scale"].items() if v < 0][:6])
    c = AR.choose_c(mx)
    r = copy.deepcopy(model); AR.apply_edits(r, plan, c)
    cos = []
    with torch.no_grad():
        for x in big:
            a = fwd(model, x).double().flatten(); b = fwd(r, x).double().flatten()
            cos.append(float(a @ b / (a.norm() * b.norm())))
    print("c", c, "fp32 등가성 코사인 최소", min(cos))
    RES.update({"ok": True, "region": plan.get("region"), "c": c, "n_scaled": len(plan["scale"]), "n_eps": len(plan["eps"]),
                "inv": [k for k, v in plan["scale"].items() if v < 0], "equiv_cos_min": min(cos), "equiv_n": len(cos)})
    RES["saved"] = {"plan": {"scale": plan["scale"], "eps": plan["eps"]}, "c": c}   # 개정 36: 구성이 다르면 이 구성으로 재측정
    r16 = copy.deepcopy(r).half(); bad = 0
    with torch.no_grad():
        for x in big:
            bad += int(not torch.isfinite(fwd(r16, x)).all())
    print("fp16 비유한", bad, "/", len(big))
    RES["fp16_nonfinite"] = bad
    # 손으로 정한 구성(1차 교정 실험)과 매개변수 단위 대조 — 바뀐 매개변수와 그 배수 지수, eps 목록 (2026-11-04)
    import math
    if which == "qwen":
        fj = json.load(open(f"eval_results/fixfam/qwen25_{'pretrained' if CKPT == 'none' else 'baseline'}.json"))["residual"]
        man = HX.apply_residual(model, lambda m: m.blocks, fj["entry"], c)
        man_eps = set(fj.get("eps_norms", []))
    else:
        from probe_gemma4_vision import apply_residual_rescale
        man = apply_residual_rescale(model, c)
        # 원고 799장 구성(fidelity_rwqa.py fix)의 eps ×c²: 층마다 input_layernorm·pre_feedforward_layernorm (개정 36에서 대조에 넣음)
        man_eps = {n for n, _ in model.named_modules() if n.endswith(("input_layernorm", "pre_feedforward_layernorm"))}
    p0 = dict(model.named_parameters()); p0.update(dict(model.named_buffers()))
    p1 = dict(man.named_parameters()); p1.update(dict(man.named_buffers()))
    man_scale = {}
    for k, v in p0.items():
        if not torch.equal(v, p1[k]):
            nz = v.abs() > 0
            ratio = float((p1[k][nz].double() / v[nz].double()).mean())
            man_scale[k] = round(math.log(ratio) / math.log(c))
    auto_scale = {k: v for k, v in plan["scale"].items()}
    only_m = sorted(set(man_scale) - set(auto_scale)); only_a = sorted(set(auto_scale) - set(man_scale))
    diff_e = sorted(k for k in set(man_scale) & set(auto_scale) if man_scale[k] != auto_scale[k])
    auto_eps = set(plan["eps"])
    print("대조 매개변수: 손", len(man_scale), "자동", len(auto_scale), "손에만", only_m[:6], "자동에만", only_a[:6], "지수 다름", diff_e[:6])
    print("대조 eps: 손", len(man_eps), "자동", len(auto_eps), "손에만", sorted(man_eps - auto_eps)[:6], "자동에만", sorted(auto_eps - man_eps)[:4], "…")
    print("대조 판정:", "매개변수 같다" if not (only_m or only_a or diff_e) else "매개변수 다르다", "/", "eps 같다" if man_eps == auto_eps else "eps 다르다")
    RES.update({"same_params": not (only_m or only_a or diff_e), "same_eps": man_eps == auto_eps, "n_manual_eps": len(man_eps),
                "n_manual_params": len(man_scale)})
if _a.out:
    RES.setdefault("ok", bool(plan.get("ok"))); RES.setdefault("reason", plan.get("reason"))
    Path(_a.out).write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str)); print("저장", _a.out)
