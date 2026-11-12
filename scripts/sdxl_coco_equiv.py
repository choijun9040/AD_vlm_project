"""개정 37 ②③ — SDXL VAE 잔차 재조정의 fp32 등가성을 COCO val2017 5,000장 전체로. 2026-11-09, 결과 보기 전 등록.
cmp_sdxl_coco.py와 같은 입력(RandomCrop 256, pad_if_needed, 시드 = 번호, [-1,1]). c는 COCO train2017 측정값(인코더 계획 json·디코더 calib json).
② 인코더: 원래 fp32 대 교정 fp32 — 분포 평균 잠재의 코사인. ③ 디코더: 같은 잠재(원래 fp32 인코더)를 원래 fp32 대 교정 fp32 디코더로 복호한 출력의 코사인.
TF32 끔. 문항마다 덧붙여 끊겨도 이어 간다. venv_sd.

    /workspace/venv_sd/bin/python scripts/sdxl_coco_equiv.py --coco /dev/shm/coco/val2017 --out eval_results/cmp/sdxl_coco_equiv.json
"""
import argparse, copy, json, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_sdxl_vae_r import apply_resid_vae          # noqa: E402
from sdxl_encoder_resid import apply_plan           # noqa: E402


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--enc_plan", default="eval_results/cmp/sdxl_enc_resid_plan_coco.json")
    ap.add_argument("--dec_calib", default="eval_results/cmp/sdxl_calib_coco.json")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from torchvision import transforms
    from diffusers import AutoencoderKL
    files = sorted(Path(a.coco).glob("*.jpg"))[:a.limit]
    crop, to_t = transforms.RandomCrop(256, pad_if_needed=True), transforms.ToTensor()
    v32 = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to("cuda").eval()
    ep = json.loads(Path(a.enc_plan).read_text())
    dc = json.loads(Path(a.dec_calib).read_text())["c"]
    r32 = copy.deepcopy(v32)
    with torch.no_grad():
        apply_plan(r32, ep["saved"])                  # 인코더 (계획 json의 c)
        apply_resid_vae(r32, dc, eps=True)            # 디코더
    part = Path(a.out + ".partial.jsonl")
    rows = [json.loads(l) for l in part.read_text().splitlines() if l.strip()] if part.exists() else []
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.double().flatten(), y.double().flatten(), dim=0))
    with part.open("a") as fo, torch.no_grad():
        for i in range(len(rows), len(files)):
            torch.manual_seed(i)
            x = (to_t(crop(Image.open(files[i]).convert("RGB"))) * 2 - 1)[None].cuda()
            z0 = v32.encode(x).latent_dist.mean
            z1 = r32.encode(x).latent_dist.mean
            y0 = v32.decode(z0).sample
            y1 = r32.decode(z0).sample
            r = {"i": i, "enc_cos": cos(z0, z1), "dec_cos": cos(y0, y1), "finite": bool(torch.isfinite(z1).all() and torch.isfinite(y1).all())}
            fo.write(json.dumps(r) + "\n"); fo.flush(); rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    q = lambda v: sorted(v)[len(v) // 2]
    res = {"n": len(rows), "enc_c": ep["saved"]["c"], "dec_c": dc, "nonfinite": sum(not r["finite"] for r in rows)}
    for k in ("enc_cos", "dec_cos"):
        v = [r[k] for r in rows]
        res[k] = {"min": min(v), "median": q(v), "below_0.99999": sum(x < 0.99999 for x in v)}
    print(f"[등가성] {json.dumps(res, ensure_ascii=False)}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
