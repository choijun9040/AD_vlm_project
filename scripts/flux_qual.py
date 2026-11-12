"""개정 42 ② — FLUX.1-schnell 정성 결과: 평가 프롬프트 1~4번(번호 순)을 원본 fp32, fp16 클리핑 끔, Draw Things 공개 설정(fp16 누산 재현),
fp16 + 잔차 재조정(c = 1/8)으로 생성해 이미지와 프롬프트별 최종 잠재 상대 오차를 저장한다. 2026-11-11, 결과 보기 전 등록. 그림용.
환경: venv_gemma(torch 2.11.0+cu128) + PYTHONPATH=/workspace/pylib_diffusers041(diffusers 0.41.0). flux_rev35.py와 같은 설정(1024, 4스텝, 안내 0, 시드 = 번호, TF32 끔).

    PYTHONPATH=/workspace/pylib_diffusers041 /workspace/venv_gemma/bin/python scripts/flux_qual.py --emb eval_results/cmp/flux_prompts.pt --out eval_results/cmp/env6/flux_qual
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flux_rev35 as F
from cmp_flux import REPO, apply_resid_flux, no_diffusers_clip


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--calib", default="eval_results/cmp/flux_calib.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from huggingface_hub import snapshot_download
    from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
    import diffusers
    from PIL import Image
    local = snapshot_download(REPO, allow_patterns=["model_index.json", "scheduler/*", "transformer/*", "vae/*"])
    E = torch.load(a.emb); S0 = int(E.get("start", 0)); c = json.load(open(a.calib))["diagnosis_bf16"]["c"]
    vae = AutoencoderKL.from_pretrained(local, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    tr = FluxTransformer2DModel.from_pretrained(local, subfolder="transformer", torch_dtype=torch.bfloat16).to("cuda").eval()
    pipe = FluxPipeline.from_pretrained(local, transformer=tr, vae=vae, text_encoder=None, text_encoder_2=None, tokenizer=None, tokenizer_2=None, torch_dtype=torch.bfloat16)

    @torch.no_grad()
    def latents_of(i):
        dt = next(tr.parameters()).dtype
        return pipe(prompt_embeds=E["prompt_embeds"][i].to("cuda", dt), pooled_prompt_embeds=E["pooled"][i].to("cuda", dt), num_inference_steps=4,
                    guidance_scale=0.0, height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(S0 + i), output_type="latent").images.float().cpu()

    @torch.no_grad()
    def decode(lat):
        lat = pipe._unpack_latents(lat.cuda().float(), 1024, 1024, pipe.vae_scale_factor)
        return vae.decode(lat / vae.config.scaling_factor + vae.config.shift_factor).sample.float().cpu()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {"n": a.n, "c": c, "torch": torch.__version__, "diffusers": diffusers.__version__, "fp16_accumulation": F.HW16, "prompts": [E["prompts"][i] for i in range(a.n)], "rows": {}}

    def save(name, i, lat):
        ok = bool(torch.isfinite(lat).all())
        if ok:
            im = ((decode(lat)[0].clamp(-1, 1) + 1) / 2 * 255).round().byte().permute(1, 2, 0).numpy()
        else:
            im = torch.zeros(1024, 1024, 3, dtype=torch.uint8).numpy()
        Image.fromarray(im).resize((384, 384), Image.LANCZOS).save(out / f"{name}_{i}.png")
        return ok

    tr.float(); ref = []
    for i in range(a.n):
        l = latents_of(i); ref.append(l); save("fp32", i, l)
    def run(name, setup, teardown):
        rows = []
        tr.half()
        with no_diffusers_clip():
            st = setup()
            for i in range(a.n):
                l = latents_of(i); ok = save(name, i, l)
                rows.append({"i": i, "finite": ok, "rel": float((l.double() - ref[i].double()).norm() / ref[i].double().norm()) if ok else None})
            teardown(st)
        res["rows"][name] = rows; print(f"[{name}] {rows}", flush=True)
        (out / "flux_qual.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    run("fp16_noclip", lambda: None, lambda st: None)
    def _rm(st):
        for h in st[0]:
            h.remove()
        if st[1] is not None:
            tr.set_attn_processor(st[1])
    run("drawthings", lambda: F.apply_dt_engine(tr), _rm)
    run("resid", lambda: apply_resid_flux(tr, c), lambda st: apply_resid_flux(tr, 1.0 / c))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
