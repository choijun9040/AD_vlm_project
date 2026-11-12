"""개정 37 덧붙임 — DTAttn 구현 확인: 변환기 1회 호출 출력으로 DTAttn(누산 fp32로 되돌림) 대 diffusers 처리기(fp16). 2026-11-10, 결과 보기 전 등록.
기준: 상대 차 ≤ 1e-2면 구현 정상(fp16 RMSNorm·RoPE·점수 반올림 차이만). 넘으면 (b2)를 쓰지 않고 고쳐서 다시 잰다.

    /workspace/venv_sd/bin/python scripts/flux_attn_check.py --emb eval_results/cmp/flux_prompts.pt --out eval_results/cmp/flux_attn_check.json
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flux_rev35 as F35                                   # noqa: E402
from cmp_flux import REPO                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--emb", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline, FluxTransformer2DModel
    local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
    E = torch.load(a.emb)
    tr = FluxTransformer2DModel.from_pretrained(local, subfolder="transformer", torch_dtype=torch.float16).to("cuda").eval()
    pipe = FluxPipeline.from_pretrained(local, transformer=tr, vae=None, text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None, torch_dtype=torch.float16)
    # 넘치지 않는 조건에서 비교한다 — (b)와 같이 Draw Things 공개 설정의 처리(잔차 fp32)를 건 상태(2026-11-10: 맨 fp16은 FLUX가 넘쳐 기준 쪽이 NaN이 된다)
    hs_dt = F35.apply_dt_flux(tr)
    cap = []
    h = tr.register_forward_pre_hook(lambda _m, _a, k: cap.append({kk: (v.detach().clone() if torch.is_tensor(v) else v) for kk, v in k.items()}),
                                     with_kwargs=True)
    with torch.no_grad(), F35.no_diffusers_clip():
        pipe(prompt_embeds=E["prompt_embeds"][0].cuda().half(), pooled_prompt_embeds=E["pooled"][0].cuda().half(), num_inference_steps=4,
             guidance_scale=0.0, height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(0), output_type="latent")
    h.remove()
    fwd = lambda x: tr(**{k: v for k, v in x.items() if k != "return_dict"}, return_dict=False)[0].double()
    res = {"n_calls": len(cap)}
    keep = F35.mm16
    F35.mm16 = lambda x, y: torch.matmul(x, y)               # 누산 fp32로 되돌림 — 처리기 구현만 비교
    rels = []
    with torch.no_grad(), F35.no_diffusers_clip():
        old = tr.attn_processors
        for x in cap:
            tr.set_attn_processor(dict(old)); y0 = fwd(x)       # set_attn_processor가 넘긴 사전을 비운다 — 매번 사본
            tr.set_attn_processor(F35.DTAttn()); y1 = fwd(x)
            rels.append(float((y1 - y0).norm() / y0.norm()))
        tr.set_attn_processor(dict(old))
    F35.mm16 = keep
    for hh in hs_dt:
        hh.remove()
    res.update({"condition": "Draw Things 공개 설정 처리(잔차 fp32) 위, fp16", "rel_per_call": rels, "rel_max": max(rels), "ok": max(rels) <= 1e-2})
    print(f"[구현 확인] 1회 호출 상대 차 {rels} · 기준 1e-2 {'통과' if res['ok'] else '미달'}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
