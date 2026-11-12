"""개정 23 — RealWorldQA 평가용 문항 중 원본 해상도 처리 형상이 952×1456인 것(465문항)의 이미지를 PNG로 푼다(디코딩한 화소 그대로 — 무손실).
보드의 Pillow가 WebP를 읽지 못할 수 있어서이고, A100 기준·보드 측정이 같은 PNG 파일을 쓴다. 2026-11-04.

    /workspace/venv_main/bin/python scripts/prep_rwqa_orin_images.py --out /dev/shm/rwqa952
"""
import argparse, json, random
from pathlib import Path
import numpy as np
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

SEED, N_CAL, SHAPE = 0, 100, (952, 1456)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--q", default="data/realworldqa/questions.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.q)]
    order = list(range(len(rows))); random.Random(SEED).shuffle(order)
    ev = sorted(order[N_CAL:])                                    # eval_realworldqa.py와 같은 나누기
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    names, idxs = [], []
    for i in ev:
        r = rows[i]
        if smart_resize(r["height"], r["width"], factor=28, min_pixels=3136, max_pixels=12845056) != SHAPE:
            continue
        im = Image.open(r["image"]).convert("RGB")
        n = f"rwqa_{i:04d}.png"
        im.save(out / "images" / n)
        assert np.array_equal(np.asarray(Image.open(out / "images" / n).convert("RGB")), np.asarray(im))   # 무손실 확인
        names.append(n); idxs.append(i)
    (out / "img465.txt").write_text("\n".join(names) + "\n")
    (out / "paths465.txt").write_text("\n".join(str(out / "images" / n) for n in names) + "\n")
    (out / "idx465.json").write_text(json.dumps(idxs))
    print(f"저장: {out} — {len(names)}장 (형상 {SHAPE})")


if __name__ == "__main__":
    main()
