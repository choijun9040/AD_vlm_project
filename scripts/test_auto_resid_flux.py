"""(R) 자동 도구 시험 — FLUX.1-schnell 변환기(개정 18 ③, 2026-11-04 사용자 「FLUX에서 자동 도구 시험」).
진단(bf16, 평가 프롬프트 2개 × 4스텝의 변환기 입력 8개, 1024×1024) → 목표 초과 연산 전부를 출발점 → 작은 입력(128×128, 텍스트 16토큰)으로 그래프 기록 → 영역
→ 손 구성(cmp_flux.apply_resid_flux)과 매개변수 단위 대조(같은 모델에 손 구성을 적용하고 매개변수 절댓값 합의 비로 배수 지수를 구한다 — fp32 사본 둘은 GPU에 못 올린다).
구성이 같으면 손 구성의 fp32 등가성(D*, 코사인 0.99999999999994)이 그대로 자동 구성의 것이다.

    /workspace/venv_sd/bin/python scripts/test_auto_resid_flux.py
개정 36 (2026-11-09): --emb eval_results/cmp/flux_prompts_calib.pt --n 100 --out <json> — 진단 입력을 c와 같은 측정용 프롬프트 100개(101~200번, 시드 = 번호) × 4스텝으로.
옵션이 없으면 예전 시험 그대로(평가 프롬프트 2개).
"""
import json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_resid as AR
import headroom_ext as HX
from cmp_flux import REPO, apply_resid_flux

from huggingface_hub import snapshot_download
from diffusers import FluxPipeline, FluxTransformer2DModel

local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
import argparse
_ap = argparse.ArgumentParser(allow_abbrev=False)
_ap.add_argument("--emb", default="eval_results/cmp/flux_prompts.pt"); _ap.add_argument("--n", type=int, default=2); _ap.add_argument("--out", default=None)
_a = _ap.parse_args()
E = torch.load(_a.emb)
S0 = int(E.get("start", 0))                                   # 시드 = 프롬프트 번호(측정용 101~200번은 100~199)
RES = {"emb": _a.emb, "n_prompts": _a.n}
tr = FluxTransformer2DModel.from_pretrained(local, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda").eval()
pipe = FluxPipeline.from_pretrained(local, transformer=tr, vae=None, text_encoder=None, text_encoder_2=None,
                                    tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)
cap = []
h = tr.register_forward_pre_hook(lambda _m, a, k: cap.append({kk: (v.detach().clone() if torch.is_tensor(v) else v) for kk, v in k.items()}),
                                 with_kwargs=True)


def run(i, size, ntok=None):
    pe = E["prompt_embeds"][i].to("cuda", torch.bfloat16)
    if ntok:
        pe = pe[:, :ntok]
    with torch.no_grad():
        pipe(prompt_embeds=pe, pooled_prompt_embeds=E["pooled"][i].to("cuda", torch.bfloat16), num_inference_steps=4,
             guidance_scale=0.0, height=size, width=size, generator=torch.Generator("cuda").manual_seed(S0 + i), output_type="latent")


for i in range(_a.n):
    run(i, 1024)
big = cap[:]; cap.clear()
run(0, 128, ntok=16)
small = cap[0]; cap.clear()
h.remove()
print("진단 입력", len(big), "추적 입력 토큰", small["hidden_states"].shape, small["encoder_hidden_states"].shape, flush=True)

fwd = lambda m, x: m(**{k: v for k, v in x.items() if k != "return_dict"}, return_dict=False)[0]
ops = HX.profile_ops(tr, big, fwd, 65504.0, 2.0)
over = [r for r in ops["rows"] if r["max_worst"] > 65504.0 / 2 and r["max_worst"] < 1e30]
mx = max(r["max_worst"] for r in over)
print("목표 초과 연산", len(over), "최댓값", mx, [r["point"] for r in over][:8], flush=True)
RES.update({"n_diag": len(big), "n_over_target": len(over), "max_worst": mx})
del ops; torch.cuda.empty_cache()

G = AR.trace(tr, lambda m: fwd(m, small))
tg, miss = AR.targets_from_points(G, [r["point"] for r in over])
print("그래프 연산", len(G["nodes"]), "출발점", len(tg), "못 찾음", miss[:5], flush=True)
RES.update({"graph_ops": len(G["nodes"]), "targets": len(tg), "missing": miss})
plan = AR.solve(G, targets=tg)
del G; torch.cuda.empty_cache()
show = {k: (v if not isinstance(v, list) or len(v) < 12 else v[:4] + [f"… 총 {len(v)}개"]) for k, v in plan.items() if k not in ("scale", "eps")}
print(json.dumps(show, ensure_ascii=False, indent=1), flush=True)
if not plan.get("ok"):
    RES.update({"ok": False, "reason": plan.get("reason")})
    if _a.out:
        Path(_a.out).write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str))
    sys.exit(0)
c = AR.choose_c(mx)
print("c", c, "eps", len(plan["eps"]), "배수 매개변수", len(plan["scale"]), "×1/c:", [k for k, v in plan["scale"].items() if v < 0][:6], flush=True)

# 손 구성과 대조 — 같은 모델에 손 구성을 적용하고 매개변수 절댓값 합의 비로 지수를 구한다(c는 2의 거듭제곱이라 bf16에서도 정확)
params = dict(tr.named_parameters())
before = {n: float(p.detach().double().abs().sum()) for n, p in params.items()}
mods = dict(tr.named_modules())
eps0 = {n: m.eps for n, m in mods.items() if isinstance(m, torch.nn.LayerNorm)}
apply_resid_flux(tr, c)
man_scale = {}
for n, p in params.items():
    a = float(p.detach().double().abs().sum())
    if before[n] > 0 and a != before[n]:
        man_scale[n] = round(math.log(a / before[n]) / math.log(c))
man_eps = {n for n, e in eps0.items() if mods[n].eps != e}
auto_scale = plan["scale"]; auto_eps = set(plan["eps"])
only_m = sorted(set(man_scale) - set(auto_scale)); only_a = sorted(set(auto_scale) - set(man_scale))
diff_e = sorted(k for k in set(man_scale) & set(auto_scale) if man_scale[k] != auto_scale[k])
print("대조 매개변수: 손", len(man_scale), "자동", len(auto_scale), "손에만", only_m[:6], "자동에만", only_a[:6], "지수 다름", diff_e[:6])
print("대조 eps: 손", len(man_eps), "자동", len(auto_eps), "손에만", sorted(man_eps - auto_eps)[:6], "자동에만", sorted(auto_eps - man_eps)[:6])
print("대조 판정:", "매개변수 같다" if not (only_m or only_a or diff_e) else "매개변수 다르다", "/", "eps 같다" if man_eps == auto_eps else "eps 다르다")
RES.update({"ok": True, "region": plan.get("region"), "c": c, "n_scaled": len(plan["scale"]), "n_eps": len(plan["eps"]),
            "inv": [k for k, v in plan["scale"].items() if v < 0], "same_params": not (only_m or only_a or diff_e), "same_eps": man_eps == auto_eps,
            "n_manual_params": len(man_scale), "n_manual_eps": len(man_eps)})
RES["saved"] = {"plan": {"scale": plan["scale"], "eps": plan["eps"]}, "c": c}   # 개정 36: 구성이 다르면 이 구성으로 재측정
if _a.out:
    Path(_a.out).write_text(json.dumps(RES, ensure_ascii=False, indent=1, default=str)); print("저장", _a.out)
