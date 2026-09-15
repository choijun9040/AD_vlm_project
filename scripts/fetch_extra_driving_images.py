"""
데이터 축 확장 — 주행 이미지 출처를 둘에서 넷으로 (8장 한계 4번).

지금까지 활성 프로파일은 nuScenes(DriveLM)와 CODA-LM 두 출처에서만 측정했다.
"활성 크기는 입력 내용이 아니라 모델 성질"을 뒷받침하려면 출처가 더 필요하다.
**학습이 필요 없다** — 이미지를 받아 프로파일만 돌리면 된다.

받는 것 (전부 공개, 다운로드 가능한 것만):
  - **BDD100K**    미국 도심·고속도로, 다양한 날씨·시간대. nuScenes와 지리·센서가 다르다.
  - **KITTI**      독일 카를스루에, 2012년 촬영. 카메라 세대가 가장 다르다.
  - **Cityscapes** 독일 도심 (샘플 공개분).

**토큰 수를 반드시 함께 기록한다.** CODA-LM이 ~720p로 재인코딩돼 1,196토큰인 반면
nuScenes는 1,836토큰이라, 활성 비교 시 해상도가 아니라 **토큰 수**를 통제해야 한다
(CLAUDE.md 참조). 이 스크립트는 원본을 그대로 저장하고 프로파일 단계에서 통제한다.

실행:
    python scripts/fetch_extra_driving_images.py --source bdd100k --limit 250
"""

import argparse
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# 인증 없이 받을 수 있는 경로만 쓴다. 접근이 막히면 그 사실을 기록하고 넘어간다.
SOURCES = {
    "bdd100k": {
        "note": "미국 도심·고속도로 (Berkeley DeepDrive)",
        "hf": ("dgural/bdd100k", "data"),
    },
    "kitti": {
        "note": "독일 카를스루에, 2012년 (KITTI) — 카메라 세대가 가장 다르다",
        "hf": ("nateraw/kitti", None),
    },
    "cityscapes": {
        "note": "독일 도심 (Cityscapes 샘플)",
        "hf": ("Chris1/cityscapes", None),
    },
}


def via_hf(repo, limit, out, subdir=None):
    """HF datasets로 받아 이미지만 뽑는다. 실패하면 예외를 올려 호출부가 기록한다."""
    from datasets import load_dataset
    ds = load_dataset(repo, split="train", streaming=True)
    n = 0
    for rec in ds:
        img = None
        for k, v in rec.items():
            if hasattr(v, "save"):          # PIL.Image
                img = v
                break
        if img is None:
            continue
        img.convert("RGB").save(out / f"{n:05d}.jpg", quality=95)
        n += 1
        if n >= limit:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--out_root", default="data/extra_driving")
    args = ap.parse_args()

    spec = SOURCES[args.source]
    out = Path(args.out_root) / args.source
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("*.jpg")))
    if have >= args.limit:
        print(f"[{args.source}] 이미 {have}장 — 건너뜀")
        return

    print(f"[{args.source}] {spec['note']}")
    repo, sub = spec["hf"]
    print(f"  HF: {repo}")
    try:
        n = via_hf(repo, args.limit, out, sub)
        print(f"  저장 {n}장 → {out}")
        if n < args.limit:
            print(f"  ⚠ 목표 {args.limit}장에 못 미침 — 분석 시 표본 수를 명시할 것")
    except Exception as e:
        print(f"  실패: {type(e).__name__}: {e}")
        print(f"  → 이 출처는 공개 경로로 받을 수 없다. 논문에 그 사실을 적는다.")
        (out / "_FAILED.txt").write_text(f"{type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
