"""개정 32 — SDXL-VAE 인코더의 (R)을 자동 도구로 구성한다. 2026-11-06, 결과 보기 전 등록(12:09 KST).
측정용: SDXL 측정용 생성 잠재(PartiPrompts 101~200번) 100개 → 원래 디코더 fp32 → RandomCrop 256(시드 = 번호). COCO와 겹치지 않는다.
진단(bf16) → 목표 초과 연산 전부 → 작은 입력 그래프 → 영역 → c → fp32 등가성(잠재 평균 코사인 ≥ 0.99999) → fp16 비유한. 계획을 json으로 저장(cmp_sdxl_coco.py --enc_plan이 쓴다). venv_sd.

    /workspace/venv_sd/bin/python scripts/sdxl_encoder_resid.py --latents eval_results/cmp/sdxl_calib.latents.pt --out eval_results/cmp/sdxl_enc_resid_plan.json
개정 33: --images 폴더(COCO train2017 100장)를 측정용으로 — 평가와 같은 RandomCrop 256(시드 = 번호).
"""
import argparse, copy, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_resid as AR
import headroom_ext as HX

FP16_MAX, TARGET = 65504.0, 2.0


class EncWrap(torch.nn.Module):
    """인코더 + quant_conv — 출력은 잠재 분포의 모멘트(평균·로그분산)."""
    def __init__(self, vae):
        super().__init__()
        self.enc, self.qc = vae.encoder, vae.quant_conv

    def forward(self, x):
        return self.qc(self.enc(x))


def apply_plan(vae, plan):
    """VAE의 인코더 쪽에 저장한 계획을 적용한다(가중치 ×c^지수, eps ×c²)."""
    with torch.no_grad():
        AR.apply_edits(EncWrap(vae), plan["plan"], plan["c"])
    return vae


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--latents", default=None)
    ap.add_argument("--images", default=None, help="개정 33: 측정용 사진 폴더(*.jpg, 이름 순)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from diffusers import AutoencoderKL
    from torchvision import transforms
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).cuda().eval()
    assert (a.latents is None) != (a.images is None), "--latents 또는 --images 하나만"
    crop = transforms.RandomCrop(256, pad_if_needed=True)
    crops = []
    if a.images:
        from PIL import Image
        to_t = transforms.ToTensor()
        for i, f in enumerate(sorted(Path(a.images).glob("*.jpg"))):
            torch.manual_seed(i)
            crops.append((to_t(crop(Image.open(f).convert("RGB"))) * 2 - 1)[None])   # [-1,1], cmp_sdxl_coco.py와 같다
    else:
        lat = torch.load(a.latents)
        with torch.no_grad():
            for i, z in enumerate(lat):
                img = vae.decode(z.cuda()).sample.clamp(-1, 1)[0].cpu()        # [-1,1], 3×1024×1024
                torch.manual_seed(i)
                crops.append(crop(img)[None])
    print(f"[측정용] {len(crops)}장 {tuple(crops[0].shape)}", flush=True)
    res = {"n_calib": len(crops)}
    wb = copy.deepcopy(EncWrap(vae)).to(torch.bfloat16)
    fwd = lambda m, x: m(x.cuda().to(next(m.parameters()).dtype))
    with torch.no_grad():
        ops = HX.profile_ops(wb, crops, fwd, FP16_MAX, TARGET)
    over = [r for r in ops["rows"] if FP16_MAX / TARGET < r["max_worst"] < 1e30]
    g = ops["global_max"]
    res["diagnosis"] = {"max_worst": g["max_worst"], "point": g["point"], "headroom_worst": g["headroom_worst"],
                        "any_exceed_rate": ops.get("any_exceed_rate"), "n_over_target": len(over),
                        "over_points": [(r["point"], r["max_worst"]) for r in sorted(over, key=lambda r: -r["max_worst"])][:30]}
    print(f"[진단] 최대 {g['max_worst']:,.0f} @ {g['point']} · 최악 여유 {g['headroom_worst']:.3f} · 초과 비율 {ops.get('any_exceed_rate')} · 목표 초과 연산 {len(over)}", flush=True)
    del wb; torch.cuda.empty_cache()
    if not over:
        res["ok"] = False; res["reason"] = "목표 초과 연산 없음"
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1)); return
    w32 = EncWrap(vae)
    G = AR.trace(w32, lambda m: m(torch.randn(1, 3, 64, 64, device="cuda")))
    tg, miss = AR.targets_from_points(G, [r["point"] for r in over])
    print(f"[그래프] 연산 {len(G['nodes'])} · 출발점 {len(tg)} · 못 찾음 {miss[:5]}", flush=True)
    plan = AR.solve(G, targets=tg)
    del G; torch.cuda.empty_cache()
    res["solve"] = {k: v for k, v in plan.items() if k not in ("scale", "eps")}
    print(f"[영역] {json.dumps(res['solve'], ensure_ascii=False)[:600]}", flush=True)
    if not plan.get("ok"):
        res["ok"] = False; res["reason"] = plan.get("reason")
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1)); return
    c = AR.choose_c(max(r["max_worst"] for r in over))
    saved = {"plan": {"scale": plan["scale"], "eps": plan["eps"]}, "c": c}
    r32 = copy.deepcopy(vae); apply_plan(r32, saved)
    cos, bad = [], 0
    with torch.no_grad():
        r16 = copy.deepcopy(r32).half()
        for x in crops:
            a0 = vae.encode(x.cuda()).latent_dist.mean.double().flatten()
            a1 = r32.encode(x.cuda()).latent_dist.mean.double().flatten()
            cos.append(float(a0 @ a1 / (a0.norm() * a1.norm())))
            bad += int(not torch.isfinite(r16.encode(x.cuda().half()).latent_dist.mean).all())
    res.update({"ok": min(cos) >= 0.99999, "c": c, "eps": sorted(plan["eps"]), "n_scaled": len(plan["scale"]),
                "scale_inv": [k for k, v in plan["scale"].items() if v < 0], "equivalence_cos_min": min(cos), "fp16_nonfinite_calib": bad})
    print(f"[결과] c = {c} · 매개변수 {len(plan['scale'])} · eps {len(plan['eps'])} · ×1/c {res['scale_inv'][:4]} · fp32 등가성 코사인 최소 {min(cos)} · fp16 비유한 {bad}/{len(crops)}", flush=True)
    res["saved"] = saved
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
