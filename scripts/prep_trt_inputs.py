"""TensorRT 검사용 입력을 미리 전처리해 .npz로 저장한다.

TRT 전용 venv에는 transformers가 없다(기존 환경을 건드리지 않기 위해 분리했다).
그래서 **A100 프로파일과 완전히 같은 전처리**를 여기서 해 두고 배열만 넘긴다.
전처리가 달라지면 붕괴율 비교가 무의미해지므로 같은 processor 설정을 쓴다.

실행:
    python scripts/prep_trt_inputs.py --limit 250 --max_pixels 1440000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent))
from dataloader import build_token_to_images

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--out", default="trt_inputs.npz")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(BASE, max_pixels=args.max_pixels,
                                         min_pixels=args.min_pixels)
    tok = build_token_to_images(VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:args.limit]
    print(f"[전처리] {len(paths)}장, max_pixels={args.max_pixels}")

    arrs, shape = {}, None
    for i, p in enumerate(paths):
        enc = proc.image_processor(images=[Image.open(p).convert("RGB")],
                                   return_tensors="np")
        pv = enc["pixel_values"].astype(np.float32)
        if shape is None:
            shape = pv.shape
            print(f"  형상 {shape}  grid={enc['image_grid_thw'].tolist()}")
        elif pv.shape != shape:
            print(f"  [건너뜀] {Path(p).name}: 형상 {pv.shape} ≠ {shape}")
            continue
        arrs[f"{i:04d}"] = pv
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)

    np.savez(args.out, **arrs)
    mb = Path(args.out).stat().st_size / 2**20
    print(f"저장: {args.out}  ({len(arrs)}장, {mb:,.0f} MiB)")
    if len(arrs) < len(paths):
        print(f"  주의: 형상 불일치로 {len(paths)-len(arrs)}장 제외 — "
              f"고정 형상 엔진이라 불가피하다. 비교 시 표본 수를 명시할 것.")


if __name__ == "__main__":
    main()
