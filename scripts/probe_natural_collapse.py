"""자연 발생 모델들의 fp16 붕괴율 — 검출 곡선에 얹을 '합성이 아닌' 점들.

**왜 필요한가.** `scripts/detection_curve.py`가 만드는 여유-붕괴 곡선은 우리가
`baseline_v2`의 MLP 가중치를 직접 줄여 만든 **합성 연속체**다. "자기가 만든 교란으로
자기 임계값을 보정했다"는 반론이 바로 성립한다. 공개 모델들은 여유가 1.7배에서
50배까지 자연스럽게 흩어져 있으므로, 같은 조건에서 **실제 fp16 붕괴율**만 재면
합성 곡선 위에 자연 발생 점을 얹어 대조할 수 있다.

두 곡선이 겹치면 보정이 합성 교란의 산물이 아님을 보인다. 어긋나면 **그 사실이
결과**다 — 가중치를 줄여 만든 여유와 학습으로 생긴 여유가 다르게 행동한다는 뜻이고,
그때는 도구의 임계값을 자연 발생 점 쪽에 맞춰야 한다.

**사전 등록 (결과 보기 전, 2026-09-15).**
  - 자연 점이 합성 곡선 위에 얹히면 → 검출 곡선의 임계값을 그대로 채택한다.
  - 자연 점만 더 일찍(= 더 큰 여유에서) 붕괴하면 → **합성 곡선이 낙관적**이라는
    뜻이므로 임계값을 자연 점 기준으로 올린다. 도구에 불리하지만 그대로 쓴다.
  - 자연 점이 더 늦게 붕괴하면 → 합성 곡선이 보수적이다. **임계값을 내리지 않는다**
    (안전 쪽 오차는 유지한다)는 점을 명시하고, 보수성의 크기만 보고한다.

실행:
    python scripts/probe_natural_collapse.py
"""
import argparse
import gc
import json
import math
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images

DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
FP16_MAX = 65504.0

# (태그, 저장소, 캐시 유지 여부) — 파이프라인 기본 모델은 지우지 않는다.
TARGETS = [
    ("base_3b", "Qwen/Qwen2.5-VL-3B-Instruct", True),
    ("base_7b", "Qwen/Qwen2.5-VL-7B-Instruct", True),
    ("qwen2vl_2b_awq", "Qwen/Qwen2-VL-2B-Instruct-AWQ", False),
    ("qwen2vl_7b_awq", "Qwen/Qwen2-VL-7B-Instruct-AWQ", False),
    ("smolvlm_instruct", "HuggingFaceTB/SmolVLM-Instruct", False),
]


def pctl(sv, q):
    return sv[min(len(sv) - 1, int(round(q * (len(sv) - 1))))]


def measure(repo, paths, max_pixels):
    """fp16으로 타워를 올려 이미지별 붕괴 여부와 마지막 블록 max를 잰다."""
    import headroom_guard as HG
    from transformers import AutoProcessor

    tower = HG.load_tower_repo(repo, "float16")
    blocks = HG.get_blocks(tower)

    proc = AutoProcessor.from_pretrained(repo)
    ip = getattr(proc, "image_processor", None)
    if max_pixels and hasattr(ip, "max_pixels"):
        ip.max_pixels, ip.min_pixels = max_pixels, 3136
        cond = f"max_pixels={max_pixels:,}"
    else:
        cond = "모델 기본값 (이 계열엔 max_pixels 설정이 없다)"
    print(f"  해상도 조건: {cond}", flush=True)

    rec = {}
    h = blocks[-1].register_forward_hook(
        lambda _m, _i, o: rec.__setitem__(
            "t", (o[0] if isinstance(o, tuple) else o).detach()))

    n_collapse, blk_max = 0, []
    with torch.no_grad():
        for p in paths:
            enc = proc(text=["<image>"], images=[Image.open(p).convert("RGB")],
                       return_tensors="pt")
            pv = enc["pixel_values"].to("cuda", torch.float16)
            grid = enc.get("image_grid_thw")
            grid = grid.to("cuda") if grid is not None else None
            out = HG.tower_forward(tower, pv, grid)
            if not torch.isfinite(out).all():
                n_collapse += 1
            t = rec["t"].float()
            blk_max.append(t.abs().max().item() if torch.isfinite(t).all()
                           else float("inf"))
    h.remove()
    del tower
    gc.collect()
    torch.cuda.empty_cache()

    finite = sorted(v for v in blk_max if math.isfinite(v))
    return {"repo": repo, "n_images": len(paths),
            "collapse_rate": n_collapse / len(paths),
            "n_nonfinite_blk": sum(1 for v in blk_max if not math.isfinite(v)),
            "blk_max_p50_fp16": pctl(finite, 0.5) if finite else None,
            "resolution": cond}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--out", default="eval_results/natural_collapse.json")
    ap.add_argument("--cleanup", action="store_true", default=True)
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.limit]
    print(f"이미지 {len(paths)}장\n")

    # 6장에서 이미 잰 bf16 여유를 붙여 (여유, 붕괴율) 쌍을 만든다.
    fam = json.loads(Path("eval_results/other_vlm_families_profile.json").read_text())

    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for tag, repo, keep in TARGETS:
        if tag in res and "collapse_rate" in res[tag]:
            print(f"[{tag}] 이미 완료 — 건너뜀"); continue
        print(f"\n{'='*70}\n[{tag}] {repo}\n{'='*70}", flush=True)
        try:
            r = measure(repo, paths, args.max_pixels)
            f = fam.get(tag, {})
            r["headroom_p50_bf16"] = f.get("headroom_p50")
            r["headroom_p95_bf16"] = f.get("headroom_p95")
            res[tag] = r
            print(f"  붕괴율 {r['collapse_rate']*100:.1f}%  "
                  f"(여유 p50 {r['headroom_p50_bf16']:.2f}배, "
                  f"p95 {r['headroom_p95_bf16']:.2f}배)", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            res[tag] = {"repo": repo, "failed": f"{type(e).__name__}: {str(e)[:200]}"}
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        if args.cleanup and not keep:
            import os
            cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            d = Path(cache) / "hub" / ("models--" + repo.replace("/", "--"))
            if d.exists():
                shutil.rmtree(d, ignore_errors=True); print(f"  캐시 정리: {d.name}")

    print("\n" + "=" * 78)
    print(f"{'모델':<22}{'여유 p50':>10}{'여유 p95':>10}{'fp16 붕괴율':>13}")
    print("-" * 78)
    for tag, r in res.items():
        if "collapse_rate" not in r:
            print(f"{tag:<22}{'실패':>10}"); continue
        print(f"{tag:<22}{r['headroom_p50_bf16']:>9.2f}배{r['headroom_p95_bf16']:>9.2f}배"
              f"{r['collapse_rate']*100:>12.1f}%")
    print("=" * 78)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
