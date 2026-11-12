"""개정 35 ①② — FLUX.1-schnell: (R) fp32 등가성을 평가 100개 전체로, Draw Things 엔진 정밀도 재현 (b2). 2026-11-09, 결과 보기 전 등록.

① 무교정 fp32 기준(4스텝 최종 잠재) — 디스크에 저장해 이어 한다.
② (R) fp32(c는 flux_calib.json의 측정용 값) 4스텝 최종 잠재와 ①의 코사인 — 개정 16의 등가성 정의를 100개로. TF32 끔.
③ (b2) = cmp_flux.apply_dt_flux(정밀도 처리 다섯 항목) + draw-things-community Flux1.swift(060f99d6)가 fp16 누산으로 표시한 연산:
   Q 투영(to_q·add_q_proj), FF 상향(ff.net.0.proj·ff_context.net.0.proj·proj_mlp), 어텐션 중간값(QKᵀ·PV fp16 누산, softmax 출력 fp16),
   q/k RMSNorm·RoPE fp16. fp16 누산은 cuBLAS(`allow_fp16_accumulation`, 해당 연산 안에서만), 없으면 K를 64씩 나눠 fp16으로 누적하는 근사.
   어텐션 배율은 q에 먼저 곱한다(구현 코드의 `.scale1` 경로 — 점수가 작아져 넘침에 유리한 쪽).
문항마다 덧붙여 끊겨도 이어 간다. venv_sd.

    /workspace/venv_sd/bin/python scripts/flux_rev35.py --emb eval_results/cmp/flux_prompts.pt --out eval_results/cmp/flux_rev35.json
"""
import argparse, contextlib, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_flux import REPO, apply_dt_flux, apply_resid_flux, no_diffusers_clip   # noqa: E402

MM = torch.backends.cuda.matmul
HW16 = hasattr(MM, "allow_fp16_accumulation")


def _acc16(on):
    if HW16:
        MM.allow_fp16_accumulation = on


def mm16(a, b):
    """fp16 누산 행렬곱 a @ b (둘 다 fp16)."""
    if HW16:
        _acc16(True)
        try:
            return torch.matmul(a, b)
        finally:
            _acc16(False)
    K, acc = a.shape[-1], None                       # 근사: K를 64씩 — 조각은 fp32로 곱하고 fp16으로 누적
    for s in range(0, K, 64):
        p = torch.matmul(a[..., s:s + 64].float(), b[..., s:s + 64, :].float()).half()
        acc = p if acc is None else acc + p
    return acc


def lin16(mod, x):
    y = mm16(x.half(), mod.weight.half().t())
    return y + mod.bias.half() if mod.bias is not None else y


def rms16(norm, x):
    """RMSNorm을 fp16에서(분산·곱 모두 fp16 출력)."""
    x = x.half()
    v = (x * x).mean(-1, keepdim=True)
    y = x * torch.rsqrt(v + norm.eps)
    return y * norm.weight.half() if getattr(norm, "weight", None) is not None else y


def rope16(x, emb):
    """x [B,H,S,D] fp16, emb = (cos, sin) [S,D] — diffusers apply_rotary_emb(use_real, unbind -1)을 fp16에서."""
    cos, sin = (t.half()[None, None] for t in emb)
    xr, xi = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    xrot = torch.stack([-xi, xr], dim=-1).flatten(3)
    return x * cos + xrot * sin


class DTAttn:
    """Draw Things 엔진의 어텐션 정밀도: Q 투영 fp16 누산, q/k RMSNorm·RoPE fp16, QKᵀ·PV fp16 누산, softmax 출력 fp16."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kw):
        B, H = hidden_states.shape[0], attn.heads
        sh = lambda t: t.view(B, -1, H, t.shape[-1] // H).transpose(1, 2)            # [B,H,S,D]
        q = sh(lin16(attn.to_q, hidden_states))
        k = sh(attn.to_k(hidden_states.half()).half())
        v = sh(attn.to_v(hidden_states.half()).half())
        q, k = rms16(attn.norm_q, q), rms16(attn.norm_k, k)
        if encoder_hidden_states is not None:
            e = encoder_hidden_states.half()
            eq = rms16(attn.norm_added_q, sh(lin16(attn.add_q_proj, e)))
            ek = rms16(attn.norm_added_k, sh(attn.add_k_proj(e).half()))
            ev = sh(attn.add_v_proj(e).half())
            q, k, v = torch.cat([eq, q], 2), torch.cat([ek, k], 2), torch.cat([ev, v], 2)
        if image_rotary_emb is not None:
            q, k = rope16(q, image_rotary_emb), rope16(k, image_rotary_emb)
        q = q * (1.0 / math.sqrt(q.shape[-1]))
        p = torch.softmax(mm16(q, k.transpose(-1, -2)), dim=-1)                       # fp16 출력
        o = mm16(p, v).transpose(1, 2).reshape(B, -1, H * q.shape[-1])
        if encoder_hidden_states is not None:
            n = encoder_hidden_states.shape[1]
            eo, o = o[:, :n], o[:, n:]
            o = attn.to_out[0](o)
            eo = attn.to_add_out(eo)
            return o, eo
        return o


def apply_dt_engine(tr):
    """(b2): apply_dt_flux + fp16 누산 연산. 되돌릴 훅·처리기를 돌려준다."""
    hs = apply_dt_flux(tr)
    old = tr.attn_processors
    tr.set_attn_processor(DTAttn())
    ffs = [b.ff.net[0].proj for b in tr.transformer_blocks] + [b.ff_context.net[0].proj for b in tr.transformer_blocks] + \
          [b.proj_mlp for b in tr.single_transformer_blocks]
    for m in ffs:                                                                    # FF 상향(linear1) — fp16 누산
        hs.append(m.register_forward_hook(lambda mod, a, o: lin16(mod, a[0])))
    return hs, old


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--calib", default="eval_results/cmp/flux_calib.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="구현 확인: 프롬프트 1개로 DTAttn(누산 fp32)과 diffusers 처리기를 비교하고 끝낸다")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
    local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
    E = torch.load(a.emb)
    N = min(a.n, len(E["prompts"]))
    S0 = int(E.get("start", 0))
    c = json.load(open(a.calib))["diagnosis_bf16"]["c"]
    vae = AutoencoderKL.from_pretrained(local, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    tr = FluxTransformer2DModel.from_pretrained(local, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda").eval()
    pipe = FluxPipeline.from_pretrained(local, transformer=tr, vae=vae, text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)

    @torch.no_grad()
    def latents_of(i):
        dt = next(tr.parameters()).dtype
        return pipe(prompt_embeds=E["prompt_embeds"][i].to("cuda", dt), pooled_prompt_embeds=E["pooled"][i].to("cuda", dt),
                    num_inference_steps=a.steps, guidance_scale=0.0, height=a.size, width=a.size,
                    generator=torch.Generator("cuda").manual_seed(S0 + i), output_type="latent").images.float().cpu()

    @torch.no_grad()
    def decode(lat):
        lat = pipe._unpack_latents(lat.cuda().float(), a.size, a.size, pipe.vae_scale_factor)
        lat = lat / vae.config.scaling_factor + vae.config.shift_factor
        return vae.decode(lat).sample.float().cpu()

    res = {"repo": REPO, "n": N, "c": c, "fp16_accumulation": "cuBLAS allow_fp16_accumulation" if HW16 else "K 64 조각 fp16 누적 근사",
           "torch": torch.__version__}
    import diffusers
    res["diffusers"] = diffusers.__version__
    print(f"[환경] torch {torch.__version__} · diffusers {diffusers.__version__} · fp16 누산 = {res['fp16_accumulation']}", flush=True)

    if a.check:                                                                       # 구현 확인(결과 아님): 누산 fp32로 두면 diffusers 처리기와 같아야 한다
        global mm16
        tr.half()
        with no_diffusers_clip():
            hs = apply_dt_flux(tr); l_ref = latents_of(0)
            old = tr.attn_processors; keep = mm16
            mm16 = lambda x, y: torch.matmul(x, y)
            tr.set_attn_processor(DTAttn()); l_my = latents_of(0)
            tr.set_attn_processor(old); mm16 = keep
            for h in hs:
                h.remove()
        rel = float((l_my.double() - l_ref.double()).norm() / l_ref.double().norm())
        print(f"[확인] DTAttn(누산 fp32) 대 diffusers 처리기 — 잠재 상대 차 {rel:.4g} (fp16 RMSNorm·RoPE 차이만)", flush=True)
        return

    out = Path(a.out)
    ref_p = Path(str(out) + ".ref32.pt")
    tr.float()
    ref = torch.load(ref_p) if ref_p.exists() else []
    for i in range(len(ref), N):                                                      # ① 기준
        ref.append(latents_of(i))
        if (i + 1) % 10 == 0 or i + 1 == N:
            torch.save(ref, ref_p); print(f"  기준 fp32 {i + 1}/{N}", flush=True)

    def partial(name):
        p = Path(str(out) + f".{name}.partial.jsonl")
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
        return p, rows

    # ② (R) fp32 등가성 — 100개 전체
    p, rows = partial("equiv")
    if len(rows) < N:
        apply_resid_flux(tr, c)
        with p.open("a") as fo:
            for i in range(len(rows), N):
                l = latents_of(i)
                r = {"i": i, "finite": bool(torch.isfinite(l).all()),
                     "cos": float(F.cosine_similarity(l.flatten().double(), ref[i].flatten().double(), dim=0)),
                     "rel": float((l.double() - ref[i].double()).norm() / ref[i].double().norm())}
                fo.write(json.dumps(r) + "\n"); fo.flush(); rows.append(r)
        apply_resid_flux(tr, 1.0 / c)                                                 # 2의 거듭제곱 — 정확히 되돌아간다
    cs = [r["cos"] for r in rows]
    res["equiv_r_fp32"] = {"n": len(rows), "nonfinite": sum(not r["finite"] for r in rows), "cos_min": min(cs), "cos_median": sorted(cs)[len(cs) // 2],
                           "rel_max": max(r["rel"] for r in rows), "below_0.99999": sum(x < 0.99999 for x in cs)}
    print(f"[(R) fp32 등가성] {json.dumps(res['equiv_r_fp32'])}", flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    # ③ fp16 구성들 — 같은 실행(같은 환경) 안에서 비교하려고 (b)·잔차 재조정도 다시 잰다(새 인스턴스, 2026-11-09 등록)
    def run_fp16(name, setup, teardown):
        p, rows = partial(name)
        if len(rows) < N:
            tr.half()
            with no_diffusers_clip():
                st = setup()
                with p.open("a") as fo:
                    for i in range(len(rows), N):
                        l = latents_of(i)
                        r = {"i": i, "finite": bool(torch.isfinite(l).all())}
                        if r["finite"]:
                            r["rel"] = float((l.double() - ref[i].double()).norm() / ref[i].double().norm())
                            im, im0 = decode(l), decode(ref[i])
                            mse = float((((im + 1) / 2).clamp(0, 1) - ((im0 + 1) / 2).clamp(0, 1)).pow(2).mean())
                            r["psnr"] = 10 * math.log10(1 / mse) if mse > 0 else float("inf")
                        fo.write(json.dumps(r) + "\n"); fo.flush(); rows.append(r)
                teardown(st)
        ok = [r for r in rows if r["finite"]]
        fe, fp = sorted(r["rel"] for r in ok), sorted(r["psnr"] for r in ok)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
        res[name] = {"nonfinite": len(rows) - len(ok), "rel_err_vs_fp32": {"median": q(fe, .5), "max": fe[-1] if fe else None},
                     "psnr_db_vs_fp32": {"median": q(fp, .5), "min": fp[0] if fp else None}}
        print(f"[{name}] {json.dumps(res[name])}", flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    def _rm(st):
        for h in st[0]:
            h.remove()
        if st[1] is not None:
            tr.set_attn_processor(st[1])
    run_fp16("b2_drawthings_engine", lambda: apply_dt_engine(tr), _rm)
    # 같은 실행 안의 비교용 — bf16과 diffusers 기본(클리핑 켬) fp16도 다시 잰다(등록 2026-11-09)
    def run_plain(name, dtype, clip):
        p, rows = partial(name)
        if len(rows) < N:
            tr.to(dtype)
            ctx = contextlib.nullcontext() if clip else no_diffusers_clip()
            with ctx, p.open("a") as fo:
                for i in range(len(rows), N):
                    l = latents_of(i)
                    r = {"i": i, "finite": bool(torch.isfinite(l).all())}
                    if r["finite"]:
                        r["rel"] = float((l.double() - ref[i].double()).norm() / ref[i].double().norm())
                        im, im0 = decode(l), decode(ref[i])
                        mse = float((((im + 1) / 2).clamp(0, 1) - ((im0 + 1) / 2).clamp(0, 1)).pow(2).mean())
                        r["psnr"] = 10 * math.log10(1 / mse) if mse > 0 else float("inf")
                    fo.write(json.dumps(r) + "\n"); fo.flush(); rows.append(r)
        ok = [r for r in rows if r["finite"]]
        fe, fp = sorted(r["rel"] for r in ok), sorted(r["psnr"] for r in ok)
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
        res[name] = {"nonfinite": len(rows) - len(ok), "rel_err_vs_fp32": {"median": q(fe, .5), "max": fe[-1] if fe else None},
                     "psnr_db_vs_fp32": {"median": q(fp, .5), "min": fp[0] if fp else None}}
        print(f"[{name}] {json.dumps(res[name])}", flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    run_plain("bf16", torch.bfloat16, True)
    run_plain("a_fp16_noclip", torch.float16, False)
    run_plain("a2_fp16_diffusers", torch.float16, True)
    run_fp16("b_drawthings", lambda: (apply_dt_flux(tr), None), _rm)
    run_fp16("c_resid", lambda: (apply_resid_flux(tr, c), None), lambda st: apply_resid_flux(tr, 1.0 / c))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
