"""공개 모델 3계열에서 INT8 지표를 재어 **지표 역전이 계열을 넘는지** 확인한다.

**왜 이 실험인가.** 9장 한계 1은 "단일 계열(Qwen2.5-VL)"이다. 기여 B(손실 설계가
여유를 움직인다)는 학습이 필요해 범위 밖이지만, **기여 A(지표 역전)는 다르다** —
스케일 불변량이 절대 크기를 볼 수 없다는 것은 **수학적 성질이지 계열 성질이 아니다.**
그런데 지금까지 INT8 지표를 우리 여섯 모델에서만 쟀으므로, 역전이 우리 계열의
특성일 수 있다는 의심이 남는다. 공개 모델 셋에서 같이 재면 그 의심이 닫힌다.

**사전 등록 (결과 보기 전, 2026-09-16).**
  - 안전한 공개 모델(여유 3.88~50.86배)의 SNR이 **우리 위험 모델보다 낮게** 나오면
    → 역전이 3계열에서 성립한다. 기여 A를 계열 독립으로 쓴다.
  - SNR이 **높게** 나오면 → 역전이 우리 계열 안에서만 성립한다는 뜻이다.
    **기여 A의 범위를 "동일 아키텍처의 통제된 변형 간"으로 좁힌다.** 불리하지만
    그대로 쓴다.
  - 순서가 뒤섞이면 → 계열 간 비교 자체가 통제되지 않았다는 뜻이다(모델 크기·학습
    데이터가 모두 다르다). **"계열 간 비교는 통제 실험이 아니다"를 명시**하고
    5변형 통제 실험만 근거로 남긴다.

**미리 적어 둘 교란.** 공개 모델들은 크기·학습 데이터·아키텍처가 모두 다르다.
5변형처럼 **손실 항 하나만 다른 통제 실험이 아니다.** 따라서 이 결과는 역전의
**일반성에 대한 정황**이지 인과가 아니다. 본문에도 그렇게 적는다.

실행:
    python scripts/probe_int8_across_families.py --limit 100
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images
from profile_int8_quant_error import int8_snr
from profile_vision_activations import percentile

DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
FP16_MAX = 65504.0

# (태그, 저장소, 캐시 유지) — 6장에서 여유를 이미 잰 공개 모델들
TARGETS = [
    ("qwen2vl_2b_awq", "Qwen/Qwen2-VL-2B-Instruct-AWQ", False),
    ("qwen2vl_7b_awq", "Qwen/Qwen2-VL-7B-Instruct-AWQ", False),
    ("smolvlm_instruct", "HuggingFaceTB/SmolVLM-Instruct", False),
]


@torch.no_grad()
def measure(repo, paths, max_pixels):
    import headroom_guard as HG
    from transformers import AutoProcessor

    tower = HG.load_tower_repo(repo, "bfloat16")
    blocks = HG.get_blocks(tower)
    depth = len(blocks)
    proc = AutoProcessor.from_pretrained(repo)
    ip = getattr(proc, "image_processor", None)
    if max_pixels and hasattr(ip, "max_pixels"):
        ip.max_pixels, ip.min_pixels = max_pixels, 3136
        cond = f"max_pixels={max_pixels:,}"
    else:
        cond = "모델 기본값"
    print(f"  depth={depth}  해상도={cond}", flush=True)

    rec = {}
    h = blocks[-1].register_forward_hook(
        lambda _m, _i, o: rec.__setitem__(
            "t", (o[0] if isinstance(o, tuple) else o).detach()))

    acc = {k: [] for k in ("max", "rms", "ratio", "snr_tensor", "snr_channel")}
    for p in paths:
        enc = proc(text=["<image>"], images=[Image.open(p).convert("RGB")],
                   return_tensors="pt")
        pv = enc["pixel_values"].to("cuda", torch.bfloat16)
        grid = enc.get("image_grid_thw")
        HG.tower_forward(tower, pv, grid.to("cuda") if grid is not None else None)
        x = rec["t"].float()
        mx = x.abs().max().item()
        rms = x.pow(2).mean().sqrt().item()
        acc["max"].append(mx)
        acc["rms"].append(rms)
        acc["ratio"].append(mx / max(rms, 1e-12))
        acc["snr_tensor"].append(int8_snr(x, per_channel=False))
        acc["snr_channel"].append(int8_snr(x, per_channel=True))
    h.remove()
    del tower
    torch.cuda.empty_cache()

    def summ(v):
        sv = sorted(t for t in v if math.isfinite(t))
        return {"p50": percentile(sv, 0.50), "p95": percentile(sv, 0.95)}
    out = {k: summ(v) for k, v in acc.items()}
    out["headroom_p50"] = FP16_MAX / out["max"]["p50"]
    out["depth"] = depth
    out["resolution"] = cond
    out["n_images"] = len(paths)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--out", default="eval_results/int8_across_families.json")
    ap.add_argument("--cleanup", action="store_true", default=True)
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.limit]
    print(f"이미지 {len(paths)}장\n")

    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for tag, repo, keep in TARGETS:
        if tag in res and "snr_tensor" in res[tag]:
            print(f"[{tag}] 이미 완료 — 건너뜀"); continue
        print(f"\n{'='*70}\n[{tag}] {repo}\n{'='*70}", flush=True)
        try:
            r = measure(repo, paths, args.max_pixels)
            res[tag] = r
            print(f"  여유 {r['headroom_p50']:.2f}배 · max/rms {r['ratio']['p50']:.1f} · "
                  f"SNR(tensor) {r['snr_tensor']['p50']:.2f} dB · "
                  f"SNR(chan) {r['snr_channel']['p50']:.2f} dB", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            res[tag] = {"repo": repo, "failed": f"{type(e).__name__}: {str(e)[:200]}"}
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        if args.cleanup and not keep:
            import shutil, os
            cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            d = Path(cache) / "hub" / ("models--" + repo.replace("/", "--"))
            if d.exists():
                shutil.rmtree(d, ignore_errors=True); print(f"  캐시 정리: {d.name}")

    print("\n" + "=" * 78)
    print(f"{'모델':<22}{'여유':>9}{'max/rms':>10}{'SNR(t)':>10}{'SNR(c)':>10}")
    print("-" * 78)
    for tag, r in res.items():
        if "snr_tensor" not in r:
            print(f"{tag:<22}{'실패':>9}"); continue
        print(f"{tag:<22}{r['headroom_p50']:>8.2f}배{r['ratio']['p50']:>10.1f}"
              f"{r['snr_tensor']['p50']:>10.2f}{r['snr_channel']['p50']:>10.2f}")
    print("=" * 78)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
