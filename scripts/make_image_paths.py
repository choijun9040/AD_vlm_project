"""
표준 순서의 DriveLM val 키프레임 이미지 경로 목록을 JSON으로 쓴다 (평가 데이터 전부로 통일, 사전 등록 4fddeb4).

지금까지의 이미지 부분집합은 전부 `build_token_to_images(v1_1_val_nus_q_only.json)` 순서의 앞쪽 `[:N]`였다.
같은 순서를 저장소 밖(다른 venv·다른 기계, 예: `bridge_measure_newhf.py --paths`)에서도 쓰도록 목록으로 뽑는다.

    python scripts/make_image_paths.py --out eval_results/val799_image_paths.json
    python scripts/make_image_paths.py --root /workspace/vlm_project_2 --out eval_results/val799_paths_ws.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from dataloader import build_token_to_images

DRIVELM_VAL_JSON = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=799, help="앞에서부터 몇 장 (기본 799 = 전부)")
    ap.add_argument("--offset", type=int, default=0, help="시작 위치")
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--root", default=None,
                    help="경로 앞부분(이 저장소의 절대 경로)을 이 값으로 바꾼다 — 다른 기계용, "
                         "예: /workspace/vlm_project_2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = build_token_to_images(str(REPO_ROOT / DRIVELM_VAL_JSON))
    paths = [str(v[args.camera]) for v in tok.values() if args.camera in v]
    paths = paths[args.offset: args.offset + args.n]
    if args.root:
        src = str(REPO_ROOT)
        bad = [p for p in paths if not p.startswith(src + "/")]
        if bad:
            raise SystemExit(f"저장소 밖 경로가 있어 --root로 바꿀 수 없다 (예: {bad[0]})")
        dst = args.root.rstrip("/")
        paths = [dst + p[len(src):] for p in paths]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(paths, indent=1))
    print(f"{len(paths)}장 ({args.camera}, offset={args.offset}) → {args.out}")
    if paths:
        print(f"  첫 경로: {paths[0]}")


if __name__ == "__main__":
    main()
