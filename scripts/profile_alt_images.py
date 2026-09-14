"""
② 데이터 일반성 — 다른 이미지 집합에서 같은 모델의 표현 범위 여유
====================================================================
지금까지의 모든 측정은 DriveLM val 키프레임의 CAM_FRONT 799장에서 이뤄졌다.
활성 크기가 **데이터 아티팩트가 아니라 모델 성질**임을 보이려면 다른 이미지 집합에서도
같은 순서·같은 여유가 나와야 한다.

다운로드 없이 쓸 수 있는 두 집합을 쓴다.
  - `CAM_BACK` : 시점이 완전히 다르다(후방). 장면 통계가 크게 다름.
  - `CAM_FRONT_LEFT` : 전방이되 화각이 다르다.

`profile_vision_activations.py`를 수정하지 않고 그 함수만 재사용한다
(λ 스윕이 그 스크립트를 호출 중이라 원본을 건드리지 않는다).

실행:
    python scripts/profile_alt_images.py --camera CAM_BACK --limit 250
"""

import argparse
import json
import random
import sys
from pathlib import Path

from transformers import AutoProcessor

sys.path.append(str(Path(__file__).parent))
import awq_compat
from profile_vision_activations import (
    BASE_3B, MODELS, build_pixel_cache, profile_model,
)

SAMPLES_DIR = "data/nuscenes/samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="CAM_BACK",
                    help="data/nuscenes/samples/ 아래 카메라 디렉터리")
    ap.add_argument("--image_dir", default=None,
                    help="임의 이미지 디렉터리 (지정 시 --camera 무시). 예: data/codalm_mini")
    ap.add_argument("--tag", default=None, help="출력 파일명 태그 (기본: 카메라명)")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--max_pixels", type=int, default=1440000, help="원본 해상도 조건")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    awq_compat.patch()

    src = Path(args.image_dir) if args.image_dir else Path(SAMPLES_DIR, args.camera)
    tag = args.tag or (Path(args.image_dir).name if args.image_dir else args.camera)
    files = sorted(p for p in src.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"{src} 에 이미지가 없다")
    random.Random(args.seed).shuffle(files)
    paths = [str(p) for p in files[: args.limit]]
    print(f"대상: {tag} {len(paths)}장 (seed={args.seed}), max_pixels={args.max_pixels}")

    processor = AutoProcessor.from_pretrained(BASE_3B, max_pixels=args.max_pixels,
                                              min_pixels=args.min_pixels)
    cache = build_pixel_cache(processor, paths, 16)
    names = [Path(p).name for p in paths]

    out_path = Path(args.out or f"eval_results/altimages_{tag}_profile.json")
    out_path.parent.mkdir(exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["_meta"] = {"source": tag, "n_images": len(paths),
                        "max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
                        "seed": args.seed}

    targets = [m for m in MODELS if (not args.only or m[0] in args.only)
               and not m[0].startswith("awq_")]
    for name, base_id, adapter, note in targets:
        r = profile_model(name, base_id, adapter, cache, names, 0)
        r["loss"] = note
        results[name] = r
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  저장: {out_path}", flush=True)

    FP16 = 65504.0
    print("\n" + "=" * 78)
    # 실제 vision token 수 — 이미지 해상도가 달라 토큰 수도 달라진다.
    # nuScenes에서 측정한 "토큰 수 → 활성" 곡선 위에 놓이는지 비교할 기준값이다.
    tokens = int(cache[0][1][0][1] * cache[0][1][0][2] // 4)
    results["_meta"]["vision_tokens"] = tokens
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"{tag} 기준 (vision token ≈ {tokens}) — 마지막 블록 p50과 fp16 여유·붕괴율")
    print(f"{'모델':22s}{'blk31 p50':>12}{'여유':>9}{'fp16 NaN':>11}")
    print("-" * 78)
    for name, *_ in targets:
        r = results.get(name)
        if not r:
            continue
        d = r["depth"]
        p50 = r["passes"]["bfloat16"]["by_layer"][str(d - 1)]["p50"]
        nan = r["passes"]["float16"]["nan_rate"] * 100
        print(f"{name:22s}{p50:>12,.0f}{FP16/p50:>8.2f}배{nan:>10.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
