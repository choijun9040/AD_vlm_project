"""
D* — FLUX.1-schnell에서 Draw Things 공개 설정과 측정 → (R)의 비교. 2026-11-04.
사전 등록: thesis_outline_20260910.md §6 개정 16 (D* 측정 조건·덧붙임, 결과 보기 전).

프롬프트 임베딩은 flux_embed.py가 미리 만든 것(PartiPrompts 앞 100개). 1024×1024, 4스텝, 안내 0, 시드 = 프롬프트 번호.
구성:
  ref32  무교정 fp32 (기준)          bf16  무교정 bf16
  a      fp16, diffusers의 ±65504 자르기를 끔(Draw Things 엔진의 조건)   a2  fp16, diffusers 그대로
  b      Draw Things 공개 설정(구현 코드 줄 단위 — 잔차 fp32, 블록 안 fp16, double 블록 17·18 FF 입력 ×⅛·출력 fp32 ×8)
  c      측정 → (R): bf16 연산 단위 크기로 c를 정하고 이미지·텍스트 두 흐름을 같은 c로(입구·쓰는 층 ×c, LayerNorm eps ×c²), fp32 등가성 확인 뒤 fp16
기록: 구성마다 비유한 출력 수, 무교정 fp32 대비 최종 잠재 상대 오차·이미지 PSNR, 측정의 이미지별 초과 비율(상실 예측).
venv_sd에서 돈다(모델은 bf16→fp16→fp32→fp16으로 제자리 변환 — 원본 체크포인트가 bf16이라 fp16·fp32 변환은 정확하다).

    /workspace/venv_sd/bin/python scripts/cmp_flux.py --emb eval_results/cmp/flux_prompts.pt --out eval_results/cmp/flux_cmp.json
"""
import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX                          # noqa: E402

REPO = "black-forest-labs/FLUX.1-schnell"
FP16_MAX, TARGET = 65504.0, 2.0
DT_BLOCKS, DT_S = (17, 18), 8                      # Draw Things 코드: upcast: i > 16 (19개 중), scaleFactor 8


def _half(_m, _i, o):
    return o.half() if torch.is_tensor(o) else o


def _float(_m, _i, o):
    return o.float() if torch.is_tensor(o) else o


def apply_dt_flux(tr):
    """Draw Things 공개 설정을 diffusers FLUX에 줄 단위로. 이식은 swift-diffusion examples/flux/main.swift(1230c66e)에서, 대조 기준은 draw-things-community Flux1.swift(060f99d6) — 다섯 항목 일치(docs/dt_code_check_20261106.md)."""
    hs = [tr.x_embedder.register_forward_hook(_float), tr.context_embedder.register_forward_hook(_float)]   # out = xEmbedder(x).to(.Float32)
    for i, b in enumerate(tr.transformer_blocks):
        for m in (b.norm1.norm, b.norm1_context.norm, b.norm2, b.norm2_context):   # xNorm(x).to(.Float16) 뒤 변조는 fp16
            hs.append(m.register_forward_hook(_half))
        if i in DT_BLOCKS:
            for lin in (b.ff.net[2], b.ff_context.net[2]):                        # (1/8)*out → outProjection → .to(.Float32)*8
                hs.append(lin.register_forward_pre_hook(lambda _m, a: (a[0] * (1.0 / DT_S),) + tuple(a[1:])))
                hs.append(lin.register_forward_hook(lambda _m, _i, o: o.float() * float(DT_S)))
    for b in tr.single_transformer_blocks:
        hs.append(b.norm.norm.register_forward_hook(_half))
    hs.append(tr.norm_out.norm.register_forward_hook(_half))                     # normFinal(out).to(.Float16)
    # diffusers의 AdaLayerNormContinuous는 시간 임베딩을 x(잔차, fp32)의 형식으로 바꿔 선형층에 넣는다 — 구현 코드는 변조(scale·shift)를 c(fp16)에서 계산한다
    hs.append(tr.norm_out.linear.register_forward_pre_hook(lambda _m, a: (a[0].half(),) + tuple(a[1:])))
    return hs


@torch.no_grad()
def apply_resid_flux(tr, c):
    """(R) — 이미지·텍스트 두 흐름을 같은 c로. 입구·쓰는 층 가중치·편향 ×c, 흐름을 읽는 LayerNorm eps ×c²(정규화가 c를 지운다)."""
    lins = [tr.x_embedder, tr.context_embedder]
    norms = [tr.norm_out.norm]
    for b in tr.transformer_blocks:
        lins += [b.attn.to_out[0], b.attn.to_add_out, b.ff.net[2], b.ff_context.net[2]]
        norms += [b.norm1.norm, b.norm1_context.norm, b.norm2, b.norm2_context]
    for b in tr.single_transformer_blocks:
        lins.append(b.proj_out)
        norms.append(b.norm.norm)
    for l in lins:
        l.weight.mul_(c)
        if l.bias is not None:
            l.bias.mul_(c)
    for n in norms:
        n.eps = n.eps * c * c
    return {"n_writers": len(lins), "n_norms": len(norms)}


@contextlib.contextmanager
def no_diffusers_clip():
    """diffusers FLUX 블록의 fp16 ±65504 자르기를 끈다 — 그 호출만 그대로 돌려준다."""
    orig = torch.Tensor.clip

    def clip(self, *a, **k):
        if a == (-65504, 65504) and not k:
            return self
        return orig(self, *a, **k)
    torch.Tensor.clip = clip
    try:
        yield
    finally:
        torch.Tensor.clip = orig


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib_only", action="store_true", help="측정(c 결정)만 하고 끝낸다 — 개정 20 측정용 프롬프트")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
    local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
    E = torch.load(a.emb)
    N = min(a.n, len(E["prompts"]))
    S0 = int(E.get("start", 0))                     # 시드 = 프롬프트 번호(0부터) — 측정용 101~200번은 시드 100~199
    vae = AutoencoderKL.from_pretrained(local, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    tr = FluxTransformer2DModel.from_pretrained(local, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda").eval()
    pipe = FluxPipeline.from_pretrained(local, transformer=tr, vae=vae, text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)
    res = {"repo": REPO, "n": N, "size": a.size, "steps": a.steps, "configs": {}}

    def latents_of(i):
        dt = next(tr.parameters()).dtype
        out = pipe(prompt_embeds=E["prompt_embeds"][i].to("cuda", dt), pooled_prompt_embeds=E["pooled"][i].to("cuda", dt),
                   num_inference_steps=a.steps, guidance_scale=0.0, height=a.size, width=a.size,
                   generator=torch.Generator("cuda").manual_seed(S0 + i), output_type="latent")
        return out.images

    @torch.no_grad()
    def decode(lat):
        lat = pipe._unpack_latents(lat.float(), a.size, a.size, pipe.vae_scale_factor)
        lat = lat / vae.config.scaling_factor + vae.config.shift_factor
        return vae.decode(lat).sample.float().cpu()

    def run_all(name, ref=None, ref_img=None):
        lats, bad, errs, psnr = [], 0, [], []
        for i in range(N):
            with torch.no_grad():
                l = latents_of(i).float()
            fin = bool(torch.isfinite(l).all())
            lats.append(l.cpu() if fin else None)
            if not fin:
                bad += 1; continue
            if ref is not None and ref[i] is not None:
                errs.append(float((l.cpu().double() - ref[i].double()).norm() / ref[i].double().norm()))
                img = decode(l)
                mse = float((((img + 1) / 2).clamp(0, 1) - ((ref_img[i] + 1) / 2).clamp(0, 1)).pow(2).mean())
                psnr.append(10 * math.log10(1 / mse) if mse > 0 else float("inf"))
        fe, fp = sorted(errs), sorted(psnr)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
        res["configs"][name] = {"nonfinite": bad, "rate": bad / N, "rel_err_vs_fp32": {"median": q(fe, .5), "max": fe[-1] if fe else None},
                                "psnr_db_vs_fp32": {"median": q(fp, .5), "min": fp[0] if fp else None}}
        print(f"[{name}] 비유한 {bad}/{N} · 잠재 상대 오차 중앙 {q(fe, .5)} · PSNR 중앙 {q(fp, .5)}", flush=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
        return lats

    # 측정 — bf16 연산 단위 크기(4스텝 전부, 이미지별 초과 비율 = 상실 예측)
    ops = HX.profile_ops(tr, list(range(N)), lambda m, i: latents_of(i), FP16_MAX, TARGET)
    g = ops["global_max"]
    # 개정 17 절차: 처음 진단의 초과 연산을 종류로 나눈다 — FF 내부(GELU 전 투영·활성)·어텐션 내부는 (R)에 불변(정규화 뒤)이라 c를 정하는 데 쓰지 않는다.
    # FLUX FF는 게이트가 없는 GELU MLP라 (G)를 등가로 쓸 수 없다 → 내부 초과가 있으면 한계로 기록한다. 어텐션 점수는 융합 커널(SDPA) 안이라 연산으로 보이지 않는다.
    inner_keys = (".ff.net.0", ".ff_context.net.0", ".proj_mlp", ".act_mlp", ".attn.norm_q", ".attn.norm_k", ".attn.norm_added_q",
                  ".attn.norm_added_k", ".attn.to_q", ".attn.to_k", ".attn.to_v", ".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj")
    is_inner = lambda pt: any(k in pt.split("::")[0] + "." for k in inner_keys) or pt.split("::")[0].endswith(".attn")
    stream_rows = [r for r in ops["rows"] if r["max_worst"] > 0 and not is_inner(r["point"])]
    inner_over = [(r["point"], r["max_worst"]) for r in ops["rows"] if is_inner(r["point"]) and r["max_worst"] > FP16_MAX / TARGET]
    smax = max((r["max_worst"] for r in stream_rows), default=0.0)
    c = HX.choose_c(smax, FP16_MAX, TARGET) if smax > FP16_MAX / TARGET else 1.0
    res["diagnosis_bf16"] = {"global_max_point": g["point"], "max_worst": g["max_worst"], "headroom_worst": g["headroom_worst"],
                             "first_exceed": ops["first_exceed"], "any_exceed_rate_predicted_loss": ops.get("any_exceed_rate"),
                             "risky_top": HX.summarize_ops(ops)["risky_top"], "c": c, "stream_max": smax,
                             "inner_over_target": sorted(inner_over, key=lambda t: -t[1])[:20],
                             "G_A_note": "FF는 GELU 단일 MLP — (G) 등가 불가; 어텐션 점수는 SDPA 안"}
    print(f"[측정] 최대 {g['max_worst']:,.0f} @ {g['point']} · 최악 여유 {g['headroom_worst']:.3f}배 · 상실 예측 {ops.get('any_exceed_rate')} · c={c}", flush=True)
    del ops; torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    if a.calib_only:
        print(f"저장: {a.out} (측정만)"); return

    # 무교정 fp32 기준과 bf16
    tr.float(); pipe.to(torch.float32)
    ref = run_all("ref32")
    ref_img = [decode(l.cuda()) if l is not None else None for l in ref]
    tr.to(torch.bfloat16)
    run_all("bf16", ref, ref_img)
    # fp16 — 자르기 끔(a), diffusers 그대로(a2), Draw Things 공개 설정(b)
    tr.half()
    with no_diffusers_clip():
        run_all("a_fp16_noclip", ref, ref_img)
    run_all("a2_fp16_diffusers", ref, ref_img)
    hs = apply_dt_flux(tr)
    run_all("b_drawthings", ref, ref_img)
    for h in hs:
        h.remove()
    # (R) — fp32에서 등가성(프롬프트 2개, 4스텝 잠재) 확인 뒤 fp16
    tr.float()
    with torch.no_grad():
        before = [latents_of(i).float().cpu() for i in range(2)]
    info = apply_resid_flux(tr, c)
    with torch.no_grad():
        after = [latents_of(i).float().cpu() for i in range(2)]
    cos = [float(torch.nn.functional.cosine_similarity(x.flatten().double(), y.flatten().double(), dim=0)) for x, y in zip(before, after)]
    res["resid"] = {"c": c, **info, "equivalence_fp32_cos": cos, "accepted": min(cos) >= 0.99999}
    print(f"[(R)] c={c} · fp32 등가성 코사인 {cos}", flush=True)
    tr.half()
    with no_diffusers_clip():
        run_all("c_resid", ref, ref_img)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
