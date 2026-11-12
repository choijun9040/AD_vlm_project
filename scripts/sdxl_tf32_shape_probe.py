"""개정 39 부속 — TF32(cudnn) 영향이 잠재 크기마다 다른지 확인. 2026-11-10 11:45 KST, 결과 보기 전 등록.
원래 SDXL VAE fp32에서 cudnn TF32만 켬·끔(행렬곱 TF32는 끔 고정)으로 같은 입력을 두 번 계산해 비트 동일 여부를 본다.
  ① 생성 잠재 5개 128×128 복호  ② 같은 잠재의 32×32 조각 복호(크기 대조)  ③ COCO val 5장 → 32×32 잠재 복호  ④ COCO 5장 인코더(256)
venv_sd.

    /workspace/venv_sd/bin/python scripts/sdxl_tf32_shape_probe.py --lat eval_results/cmp/env3/sdxl_gen_cmp.latents.pt --coco /dev/shm/coco/val2017 --out eval_results/cmp/env3/sdxl_tf32_shape_probe.json
"""
import argparse, json
from pathlib import Path
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--lat", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from torchvision import transforms
    from diffusers import AutoencoderKL
    torch.backends.cuda.matmul.allow_tf32 = False
    v = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    lat = torch.load(a.lat)[:a.n]
    files = sorted(Path(a.coco).glob("*.jpg"))[:a.n]
    crop, to_t = transforms.RandomCrop(256, pad_if_needed=True), transforms.ToTensor()
    imgs = []
    for i, f in enumerate(files):
        torch.manual_seed(i)
        imgs.append((to_t(crop(Image.open(f).convert("RGB"))) * 2 - 1)[None])

    def run(tf32, fn, xs):
        torch.backends.cudnn.allow_tf32 = tf32
        with torch.no_grad():
            return [fn(x.cuda()).float().cpu() for x in xs]

    torch.backends.cudnn.allow_tf32 = False
    with torch.no_grad():
        coco_lat = [v.encode(x.cuda()).latent_dist.mean.cpu() for x in imgs]
    cases = {"1_gen_128": (lambda z: v.decode(z).sample, lat),
             "2_gen_slice_32": (lambda z: v.decode(z).sample, [z[..., 48:80, 48:80].contiguous() for z in lat]),
             "3_coco_32": (lambda z: v.decode(z).sample, coco_lat),
             "4_coco_encoder_256": (lambda x: v.encode(x).latent_dist.mean, imgs)}
    res = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "n": a.n}
    for name, (fn, xs) in cases.items():
        on, off = run(True, fn, xs), run(False, fn, xs)
        off2 = run(False, fn, xs)                       # 같은 설정 반복 — 결정성 확인
        res[name] = {"shape_in": list(xs[0].shape), "bitwise_equal_on_off": sum(bool(torch.equal(p, q)) for p, q in zip(on, off)),
                     "max_abs_on_off": max(float((p - q).abs().max()) for p, q in zip(on, off)),
                     "repeat_equal_off": sum(bool(torch.equal(p, q)) for p, q in zip(off, off2))}
        print(f"[{name}] {json.dumps(res[name])}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
