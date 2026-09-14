"""
CODA-LM 이미지 내려받기 (② 데이터 일반성용)
=============================================
nuScenes 밖 이미지에서도 같은 활성 크기·여유가 나오는지 보려면 다른 데이터셋이 필요하다.
CODA-LM llava-format의 **Mini 스플릿**(193장, 약 143 MiB)만 받아 이미지를 추출한다.

주의: CODA는 KITTI·nuScenes·ONCE에서 구성된 데이터셋이라 **nuScenes와 완전히
분리되지 않는다.** 추출 후 id 접두어로 출처 구성을 집계해 함께 보고한다.

실행:
    python scripts/fetch_codalm_images.py
"""

import argparse
import collections
import io
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/KaiChen1998/coda-lm-llava-format/"
       "resolve/main/English/Mini-00000-of-00001.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/codalm_mini")
    ap.add_argument("--parquet", default="data/codalm_mini/_Mini.parquet")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq_path = Path(args.parquet)

    if not pq_path.exists():
        print(f"내려받는 중: {URL}")
        req = urllib.request.Request(URL, headers={"User-Agent": "fetch/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, pq_path.open("wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total % (20 << 20) < (1 << 20):
                    print(f"  {total / 2**20:.0f} MiB", flush=True)
        print(f"완료: {total / 2**20:.1f} MiB")

    import pyarrow.parquet as pq
    from PIL import Image

    t = pq.read_table(pq_path, columns=["id", "image"])
    ids = t.column("id").to_pylist()
    imgs = t.column("image").to_pylist()
    print(f"행 수: {len(ids)}")

    src = collections.Counter()
    n = 0
    for i, (rid, im) in enumerate(zip(ids, imgs)):
        b = im["bytes"] if isinstance(im, dict) else im
        if b is None:
            continue
        p = out / f"{i:04d}_{Path(str(rid)).stem}.jpg"
        if not p.exists():
            Image.open(io.BytesIO(b)).convert("RGB").save(p, quality=95)
        n += 1
        # id 접두어로 원출처 추정 (CODA는 KITTI/nuScenes/ONCE 혼합)
        s = str(rid).lower()
        for key in ("kitti", "nuscenes", "once", "test", "val", "train"):
            if key in s:
                src[key] += 1
                break
        else:
            src["기타"] += 1

    print(f"\n추출 완료: {n}장 → {out}")
    print("id 기준 출처 추정:", dict(src))
    print("예시 id:", ids[:5])


if __name__ == "__main__":
    main()
