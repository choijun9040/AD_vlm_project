"""
E-A — Hugging Face DebugUnderflowOverflow와 본 연구 진단의 비교. 2026-11-04.
사전 등록: thesis_outline_20260910.md §6 개정 15 (결과 보기 전).

transformers.debug_utils.DebugUnderflowOverflow를 fp16 비전 타워에 문서대로 걸고 799장을 한 장씩 돌린다. 도구는 inf/nan을 처음 만나면
직전 프레임(모듈별 절댓값 최소·최대)을 출력하고 ValueError로 멈춘다 — 이미지마다 그 예외를 잡아 경고 여부와 처음 inf/nan을 낸 모듈을 기록하고,
도구의 상태(프레임·감지 표시)를 비운 뒤 다음 장으로 간다(문서의 사용법은 한 번 멈추면 끝이므로, 799장 각각을 독립 실행으로 본다).

    python scripts/cmp_debug_overflow.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --tag baseline_unfixed \
        --out eval_results/cmp/ea_baseline_unfixed.json
"""
import argparse
import contextlib
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat                                   # noqa: E402
import headroom_guard as G                          # noqa: E402
from dataloader import build_token_to_images       # noqa: E402


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vision_state", default=None, help="(R) 교정 가중치(.pt) — 옆 파일 eps도 적용")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    awq_compat.patch()
    from transformers import AutoProcessor
    from transformers.debug_utils import DebugUnderflowOverflow

    tower = G.load_tower(a.checkpoint, G.BASE_3B, "float32")
    if a.vision_state:
        import headroom_ext as HX
        sd = torch.load(a.vision_state, map_location="cpu")
        fp, ep = Path(a.vision_state + ".fix.json"), Path(a.vision_state + ".eps.json")
        side = json.loads(fp.read_text()) if fp.exists() else ({"eps": json.loads(ep.read_text())} if ep.exists() else {})
        HX.load_fixed(tower, sd, side)
    tower = tower.half().eval()

    proc = AutoProcessor.from_pretrained(G.BASE_3B, max_pixels=a.max_pixels, min_pixels=a.min_pixels)
    tok = build_token_to_images(G.DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]

    dbg = DebugUnderflowOverflow(tower)              # 문서 기본값(max_frames_to_save=21)
    rows, first_mods = [], Counter()
    pat = re.compile(r"^\s*\S+\s+(\S+)\s+(\S+)")     # 「abs min · abs max · 모듈 이름 …」 프레임 줄
    for i, p in enumerate(paths):
        x = G.build_cache(proc, [p], "float16")[0]
        dbg.frames.clear(); dbg.frame = []; dbg.detected_overflow = False; dbg.total_calls = 0
        buf = io.StringIO()
        flagged, mod, nonfinite = False, None, None
        with contextlib.redirect_stdout(buf), torch.no_grad():
            try:
                out = G.tower_forward(tower, x[0], x[1])
                nonfinite = not torch.isfinite(out).all().item()
            except ValueError as e:
                if "inf/nan detected" not in str(e):
                    raise
                flagged = True
        if flagged:
            for line in buf.getvalue().splitlines():
                m = pat.match(line)
                if m and m.group(1).lower() in ("inf", "nan"):
                    mod = m.group(2)
                    break
            first_mods[mod] += 1
        rows.append({"idx": i, "image": Path(p).name, "flagged": flagged, "first_module": mod,
                     "output_nonfinite": nonfinite})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)} 경고 {sum(r['flagged'] for r in rows)}", flush=True)

    res = {"tag": a.tag, "checkpoint": a.checkpoint, "vision_state": a.vision_state, "max_pixels": a.max_pixels,
           "n": len(rows), "n_flagged": sum(r["flagged"] for r in rows),
           "n_output_nonfinite_unflagged": sum(bool(r["output_nonfinite"]) for r in rows),
           "first_module_counts": dict(first_mods.most_common())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    with open(a.out.replace(".json", "_per_image.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
