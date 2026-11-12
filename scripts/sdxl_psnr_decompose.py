"""개정 39 부속 2·3 — COCO PSNR 변동의 원인 분해. 2026-11-10 11:48 KST, 결과 보기 전 등록(개요 개정 39 아래).
참값 = 원래 SDXL VAE 디코더 fp64. 원래 fp32(TF32 끔·켬)와 잔차 재조정 디코더 fp16(SDPA 백엔드·어텐션 정밀도 변형)의 오차를 참값 대비로 분해한다.
venv_sd.

    /workspace/venv_sd/bin/python scripts/sdxl_psnr_decompose.py --coco /dev/shm/coco/val2017 --lat eval_results/cmp/env3/sdxl_gen_cmp.latents.pt \
        --c_coco 0.03125 --c_gen 0.015625 --out eval_results/cmp/env3/sdxl_psnr_decompose.json
"""
import argparse, copy, json, math, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_sdxl_vae_r import apply_resid_vae          # noqa: E402


def to01(y):
    return (y.double().clamp(-1, 1) + 1) / 2


def psnr(a, b):
    m = float((to01(a) - to01(b)).pow(2).mean())
    return 10 * math.log10(1 / m) if m > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--lat", required=True)
    ap.add_argument("--n_coco", type=int, default=50)
    ap.add_argument("--n_gen", type=int, default=10)
    ap.add_argument("--c_coco", type=float, required=True)
    ap.add_argument("--c_gen", type=float, required=True)
    ap.add_argument("--part3", action="store_true", help="부속 3: cudnn.benchmark·GroupNorm fp32·합성곱 fp32 변형")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from torchvision import transforms
    from diffusers import AutoencoderKL
    from torch.nn.attention import sdpa_kernel, SDPBackend
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    v32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    v64 = copy.deepcopy(v32).double()
    crop, to_t = transforms.RandomCrop(256, pad_if_needed=True), transforms.ToTensor()
    coco = []
    with torch.no_grad():
        for i, f in enumerate(sorted(Path(a.coco).glob("*.jpg"))[:a.n_coco]):
            torch.manual_seed(i)
            x = (to_t(crop(Image.open(f).convert("RGB"))) * 2 - 1)[None].cuda()
            coco.append(v32.encode(x).latent_dist.mean)
    gen = [z.cuda() for z in torch.load(a.lat)[:a.n_gen]]

    def rr16(c, attn32=False, gn32=False, conv32=False):
        r = copy.deepcopy(v32)
        with torch.no_grad():
            apply_resid_vae(r, c, eps=True)
        r = r.half()
        if attn32:
            at = r.decoder.mid_block.attentions[0].float()
            at.register_forward_pre_hook(lambda m, args, kw: (tuple(t.float() if torch.is_tensor(t) else t for t in args), kw), with_kwargs=True)
            at.register_forward_hook(lambda m, i, o: o.half())
        if gn32 or conv32:
            kinds = (torch.nn.GroupNorm,) if gn32 else (torch.nn.Conv2d,)
            for m in r.decoder.modules():
                if isinstance(m, kinds):
                    m.float()
                    m.register_forward_pre_hook(lambda mm, args: tuple(t.float() for t in args))
                    m.register_forward_hook(lambda mm, i, o: o.half())
        return r

    res = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "n_coco": len(coco), "n_gen": len(gen)}
    for name, lats, c in (("coco", coco, a.c_coco), ("gen", gen, a.c_gen)):
        R16, R16a = rr16(c), rr16(c, attn32=True)
        R16g, R16c = (rr16(c, gn32=True), rr16(c, conv32=True)) if a.part3 else (None, None)
        rows = []
        with torch.no_grad():
            for z in lats:
                t = v64.decode(z.double()).sample
                torch.backends.cudnn.allow_tf32 = False; off = v32.decode(z).sample
                torch.backends.cudnn.allow_tf32 = True; on = v32.decode(z).sample
                torch.backends.cudnn.allow_tf32 = False
                outs = {"rr16_default": R16.decode(z.half()).sample}
                for bn, be in (("rr16_math", SDPBackend.MATH), ("rr16_efficient", SDPBackend.EFFICIENT_ATTENTION)):
                    try:
                        with sdpa_kernel(be):
                            outs[bn] = R16.decode(z.half()).sample
                    except Exception as e:
                        outs[bn] = None; res.setdefault("errors", {})[f"{name}:{bn}"] = str(e)[:200]
                outs["rr16_attn32"] = R16a.decode(z.half()).sample
                if a.part3:
                    torch.backends.cudnn.benchmark = True
                    outs["rr16_benchmark"] = R16.decode(z.half()).sample
                    torch.backends.cudnn.benchmark = False
                    outs["rr16_gn32"] = R16g.decode(z.half()).sample
                    outs["rr16_conv32"] = R16c.decode(z.half()).sample
                er = {"ref_off_vs_true": psnr(off, t), "ref_on_vs_true": psnr(on, t)}
                for k, y in outs.items():
                    if y is None:
                        continue
                    er[f"{k}_vs_true"] = psnr(y, t); er[f"{k}_vs_ref_off"] = psnr(y, off); er[f"{k}_vs_ref_on"] = psnr(y, on)
                e_r = (to01(outs["rr16_default"]) - to01(t)).flatten(); e_on = (to01(on) - to01(t)).flatten(); e_off = (to01(off) - to01(t)).flatten()
                cs = lambda p, q: float(torch.nn.functional.cosine_similarity(p, q, dim=0))
                er["cos_err_rr16_ref_on"] = cs(e_r, e_on); er["cos_err_rr16_ref_off"] = cs(e_r, e_off)
                rows.append(er)
        med = lambda v: sorted(v)[len(v) // 2]
        res[name] = {"c": c, "median": {k: med([r[k] for r in rows if k in r]) for k in rows[0]}}
        print(f"[{name}] {json.dumps(res[name]['median'], ensure_ascii=False)}", flush=True)
        del R16, R16a, R16g, R16c; torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
