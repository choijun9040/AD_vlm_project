"""(R) 자동 도구 시험 — SDXL VAE 디코더(개정 18 ③). 진단(bf16) → 목표 초과 연산 전부를 출발점 → 영역 → c → fp32 등가성. 손으로 정한 구성(개정 16 C*)과 비교.
개정 36 (2026-11-09): --latents eval_results/cmp/sdxl_calib.latents.pt --out <json> — 진단·등가성 입력을 c와 같은 측정용 생성 잠재 100개(PartiPrompts 101~200번)로,
수동 구성(cmp_sdxl_vae_r.apply_resid_vae)과 매개변수·eps 단위 대조. 옵션이 없으면 예전 시험 그대로(DriveLM 32장)."""
import copy, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_resid as AR
import headroom_ext as HX
from dataloader import build_token_to_images
from cmp_sdxl_vae import load_img, VAL
from diffusers import AutoencoderKL

import argparse, math
_ap = argparse.ArgumentParser(allow_abbrev=False)
_ap.add_argument("n", nargs="?", type=int, default=32); _ap.add_argument("--latents", default=None); _ap.add_argument("--out", default=None)
_a = _ap.parse_args()
N = _a.n
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
if _a.latents:
    lat = [z.cuda().float() for z in torch.load(_a.latents)]
else:
    tok = build_token_to_images(VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:N]
    with torch.no_grad():
        lat = [vae.encode(load_img(p, 1024, 576).cuda()).latent_dist.mean for p in paths]
RES = {"latents": _a.latents, "n_diag": len(lat)}
vb = copy.deepcopy(vae).to(torch.bfloat16)
with torch.no_grad():
    cache = [vb.post_quant_conv(z.to(torch.bfloat16)) for z in lat]
ops = HX.profile_ops(vb.decoder, cache, lambda m, x: m(x), 65504.0, 2.0)
over = [r for r in ops["rows"] if r["max_worst"] > 65504.0 / 2 and not r["upcast"]]
mx = max(r["max_worst"] for r in over)
print("목표 초과 연산", len(over), "최댓값", mx, [r["point"] for r in over][:12])
RES.update({"n_over_target": len(over), "max_worst": mx})
del vb; torch.cuda.empty_cache()
G = AR.trace(vae.decoder, lambda m: m(torch.randn(1, 4, 16, 16, device="cuda")))
tg, miss = AR.targets_from_points(G, [r["point"] for r in over])
print("출발점", len(tg), "그래프에서 못 찾음", miss)
RES.update({"graph_ops": len(G["nodes"]), "targets": len(tg), "missing": miss})
plan = AR.solve(G, targets=tg)
print(json.dumps({k: v for k, v in plan.items() if k != "scale"}, ensure_ascii=False, indent=1))
print("배수:", json.dumps(plan.get("scale"), ensure_ascii=False))
if plan.get("ok"):
    c = AR.choose_c(mx)
    r = copy.deepcopy(vae); ch = AR.apply_edits(r.decoder, plan, c)
    cos = []
    with torch.no_grad():
        for z in lat:
            a = vae.decode(z).sample.double().flatten(); b = r.decode(z).sample.double().flatten()
            cos.append(float(a @ b / (a.norm() * b.norm())))
    print("c", c, "eps 변경 모듈", sorted(ch), "fp32 등가성 코사인 최소", min(cos))
    r16 = copy.deepcopy(r).half(); bad = 0
    with torch.no_grad():
        for z in lat:
            bad += int(not torch.isfinite(r16.decode(z.half()).sample).all())
    print("fp16 비유한", bad, "/", len(lat))
    RES.update({"ok": True, "region": plan.get("region"), "c": c, "n_scaled": len(plan["scale"]), "n_eps": len(plan["eps"]),
                "inv": [k for k, v in plan["scale"].items() if v < 0], "equiv_cos_min": min(cos), "equiv_n": len(cos), "fp16_nonfinite": bad})
    RES["saved"] = {"plan": {"scale": plan["scale"], "eps": plan["eps"]}, "c": c}   # 개정 36: 구성이 다르면 이 구성으로 재측정
    # 수동 구성과 매개변수·eps 단위 대조 (개정 36)
    from cmp_sdxl_vae_r import apply_resid_vae
    m = copy.deepcopy(vae)
    with torch.no_grad():
        apply_resid_vae(m, c, eps=True)
    p0, p1 = dict(vae.decoder.named_parameters()), dict(m.decoder.named_parameters())
    man_scale = {}
    for k, v in p0.items():
        if not torch.equal(v, p1[k]):
            nz = v.abs() > 0
            man_scale[k] = round(math.log(float((p1[k][nz].double() / v[nz].double()).mean())) / math.log(c))
    m0, m1 = dict(vae.decoder.named_modules()), dict(m.decoder.named_modules())
    man_eps = {k for k, mod in m0.items() if hasattr(mod, "eps") and isinstance(mod.eps, float) and m1[k].eps != mod.eps}
    auto_scale, auto_eps = plan["scale"], set(plan["eps"])
    only_m = sorted(set(man_scale) - set(auto_scale)); only_a = sorted(set(auto_scale) - set(man_scale))
    diff_e = sorted(k for k in set(man_scale) & set(auto_scale) if man_scale[k] != auto_scale[k])
    print("대조 매개변수: 수동", len(man_scale), "자동", len(auto_scale), "수동에만", only_m[:6], "자동에만", only_a[:6], "지수 다름", diff_e[:6])
    print("대조 eps: 수동", sorted(man_eps), "자동", sorted(auto_eps))
    RES.update({"same_params": not (only_m or only_a or diff_e), "same_eps": man_eps == auto_eps,
                "n_manual_params": len(man_scale), "n_manual_eps": len(man_eps), "only_manual": only_m, "only_auto": only_a})
if _a.out:
    RES.setdefault("ok", bool(plan.get("ok"))); RES.setdefault("reason", plan.get("reason"))
    Path(_a.out).write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str)); print("저장", _a.out)
