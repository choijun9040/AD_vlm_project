"""(b2) 가르기 진단 — 직접 구현 어텐션의 fp16 처리를 하나씩 켜며 diffusers 처리기와 1회 호출 비교. 2026-11-10, 결과 보기 전 등록.
V0 = diffusers와 같은 계산, V1 = +정규화 fp16, V2 = +RoPE fp16, V3 = +점수 fp16 반올림 후 softmax(= (b2) 어텐션, 누산 fp32). 조건: (b) 처리(잔차 fp32) 위 fp16.
판정: V0 최대 ≤ 1e-3 → 구현 정상.

    /workspace/venv_sd/bin/python scripts/flux_attn_diag.py --emb eval_results/cmp/flux_prompts.pt --out eval_results/cmp/flux_attn_diag.json
"""
import argparse, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flux_rev35 as F35                                   # noqa: E402
from cmp_flux import REPO                                  # noqa: E402


def rms_ref(norm, x):
    """diffusers RMSNorm과 같게: fp32에서 분산·정규화, 가중치 곱 뒤 입력 형식으로."""
    d = x.dtype; xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)
    if getattr(norm, "weight", None) is not None:
        y = (y.to(norm.weight.dtype) * norm.weight) if norm.weight.dtype in (torch.float16, torch.bfloat16) else y * norm.weight
    return y.to(d)


def rope_ref(x, emb):
    """diffusers apply_rotary_emb(use_real, unbind -1)과 같게: fp32에서 계산 후 입력 형식으로."""
    cos, sin = (t.float()[None, None] for t in emb)
    xr, xi = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    xrot = torch.stack([-xi, xr], dim=-1).flatten(3)
    return (x.float() * cos + xrot * sin).to(x.dtype)


class AttnV:
    def __init__(self, norm16, rope16, score16, keep_dtype=False):
        self.n16, self.r16, self.s16, self.keep = norm16, rope16, score16, keep_dtype

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, **kw):
        if self.keep:                                  # fp32 시험: 입력 형식 그대로(diffusers와 같은 계산)
            return self.call_keep(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        B, H = hidden_states.shape[0], attn.heads
        sh = lambda t: t.view(B, -1, H, t.shape[-1] // H).transpose(1, 2)
        nrm = F35.rms16 if self.n16 else rms_ref
        q, k, v = sh(attn.to_q(hidden_states.half())), sh(attn.to_k(hidden_states.half())), sh(attn.to_v(hidden_states.half()))
        q, k = nrm(attn.norm_q, q).half(), nrm(attn.norm_k, k).half()
        if encoder_hidden_states is not None:
            e = encoder_hidden_states.half()
            eq = nrm(attn.norm_added_q, sh(attn.add_q_proj(e))).half()
            ek = nrm(attn.norm_added_k, sh(attn.add_k_proj(e))).half()
            ev = sh(attn.add_v_proj(e))
            q, k, v = torch.cat([eq, q], 2), torch.cat([ek, k], 2), torch.cat([ev, v], 2)
        if image_rotary_emb is not None:
            rp = F35.rope16 if self.r16 else rope_ref
            q, k = rp(q, image_rotary_emb), rp(k, image_rotary_emb)
        if self.s16:
            q2 = q * (1.0 / math.sqrt(q.shape[-1]))
            o = torch.matmul(torch.softmax(torch.matmul(q2, k.transpose(-1, -2)), dim=-1), v)
        else:
            o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, -1, H * q.shape[-1]).to(q.dtype)
        if encoder_hidden_states is not None:
            n = encoder_hidden_states.shape[1]
            return attn.to_out[0](o[:, n:]), attn.to_add_out(o[:, :n])
        return o


def _call_keep(self, attn, hs, ehs, emb):
    B, H = hs.shape[0], attn.heads
    sh = lambda t: t.view(B, -1, H, t.shape[-1] // H).transpose(1, 2)
    q, k, v = sh(attn.to_q(hs)), sh(attn.to_k(hs)), sh(attn.to_v(hs))
    q, k = rms_ref(attn.norm_q, q), rms_ref(attn.norm_k, k)
    if ehs is not None:
        eq, ek, ev = sh(attn.add_q_proj(ehs)), sh(attn.add_k_proj(ehs)), sh(attn.add_v_proj(ehs))
        eq, ek = rms_ref(attn.norm_added_q, eq), rms_ref(attn.norm_added_k, ek)
        q, k, v = torch.cat([eq, q], 2), torch.cat([ek, k], 2), torch.cat([ev, v], 2)
    if emb is not None:
        q, k = rope_ref(q, emb), rope_ref(k, emb)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, -1, H * q.shape[-1]).to(q.dtype)
    if ehs is not None:
        n = ehs.shape[1]
        return attn.to_out[0](o[:, n:]), attn.to_add_out(o[:, :n])
    return o


AttnV.call_keep = _call_keep


def main2(a):
    """원인 찾기(같은 날 등록): ① diffusers 처리기 반복(fp16) ② fp32 모델에서 V0(입력 형식 그대로) 대 diffusers."""
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline, FluxTransformer2DModel
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
    E = torch.load(a.emb)
    res = {}
    for dt in (torch.float16, torch.float32):
        tr = FluxTransformer2DModel.from_pretrained(local, subfolder="transformer", torch_dtype=dt).to("cuda").eval()
        pipe = FluxPipeline.from_pretrained(local, transformer=tr, vae=None, text_encoder=None, text_encoder_2=None,
                                            tokenizer=None, tokenizer_2=None, torch_dtype=dt)
        hs = F35.apply_dt_flux(tr) if dt == torch.float16 else []
        cap = []
        h = tr.register_forward_pre_hook(lambda _m, _a, k: cap.append({kk: (v.detach().clone() if torch.is_tensor(v) else v) for kk, v in k.items()}),
                                         with_kwargs=True)
        with torch.no_grad(), F35.no_diffusers_clip():
            pipe(prompt_embeds=E["prompt_embeds"][0].cuda().to(dt), pooled_prompt_embeds=E["pooled"][0].cuda().to(dt), num_inference_steps=4,
                 guidance_scale=0.0, height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(0), output_type="latent")
        h.remove()
        fwd = lambda x: tr(**{k: v for k, v in x.items() if k != "return_dict"}, return_dict=False)[0].double()
        old = tr.attn_processors
        with torch.no_grad(), F35.no_diffusers_clip():
            tr.set_attn_processor(dict(old)); y0 = [fwd(x) for x in cap]
            if dt == torch.float16:
                y1 = [fwd(x) for x in cap]
                rel = [float((b - c).norm() / c.norm()) for b, c in zip(y1, y0)]
                res["repeat_fp16"] = {"rel_per_call": rel, "rel_max": max(rel)}
                print(f"[반복 fp16] {[f'{r:.3g}' for r in rel]}", flush=True)
            else:
                tr.set_attn_processor(AttnV(False, False, False, keep_dtype=True))
                rel = [float((fwd(x) - c).norm() / c.norm()) for x, c in zip(cap, y0)]
                res["v0_fp32"] = {"rel_per_call": rel, "rel_max": max(rel), "implementation_same": max(rel) <= 1e-4}
                print(f"[V0 fp32] {[f'{r:.3g}' for r in rel]} → {'같은 계산' if max(rel) <= 1e-4 else '구현 실수'}", flush=True)
                tr.set_attn_processor(dict(old))
        for hh in hs:
            hh.remove()
        del tr, pipe, cap; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--emb", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cause", action="store_true")
    a = ap.parse_args()
    if a.cause:
        return main2(a)
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline, FluxTransformer2DModel
    local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
    E = torch.load(a.emb)
    tr = FluxTransformer2DModel.from_pretrained(local, subfolder="transformer", torch_dtype=torch.float16).to("cuda").eval()
    pipe = FluxPipeline.from_pretrained(local, transformer=tr, vae=None, text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None, torch_dtype=torch.float16)
    hs = F35.apply_dt_flux(tr)
    cap = []
    h = tr.register_forward_pre_hook(lambda _m, _a, k: cap.append({kk: (v.detach().clone() if torch.is_tensor(v) else v) for kk, v in k.items()}),
                                     with_kwargs=True)
    with torch.no_grad(), F35.no_diffusers_clip():
        pipe(prompt_embeds=E["prompt_embeds"][0].cuda().half(), pooled_prompt_embeds=E["pooled"][0].cuda().half(), num_inference_steps=4,
             guidance_scale=0.0, height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(0), output_type="latent")
    h.remove()
    fwd = lambda x: tr(**{k: v for k, v in x.items() if k != "return_dict"}, return_dict=False)[0].double()
    old = tr.attn_processors
    res = {"condition": "(b) 처리(잔차 fp32) 위 fp16, 4회 호출", "variants": {}}
    V = {"V0_ref": (False, False, False), "V1_norm16": (True, False, False), "V2_rope16": (True, True, False), "V3_score16": (True, True, True)}
    with torch.no_grad(), F35.no_diffusers_clip():
        tr.set_attn_processor(dict(old)); y0 = [fwd(x) for x in cap]
        for name, fl in V.items():
            tr.set_attn_processor(AttnV(*fl))
            rel = [float((fwd(x) - y).norm() / y.norm()) for x, y in zip(cap, y0)]
            res["variants"][name] = {"rel_per_call": rel, "rel_max": max(rel)}
            print(f"[{name}] {[round(r, 6) for r in rel]}", flush=True)
        tr.set_attn_processor(dict(old))
    res["implementation_ok"] = res["variants"]["V0_ref"]["rel_max"] <= 1e-3
    print(f"[판정] V0 최대 {res['variants']['V0_ref']['rel_max']:.3g} → {'구현 정상' if res['implementation_ok'] else '구현 실수'}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
