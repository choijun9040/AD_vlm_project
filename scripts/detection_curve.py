"""headroom_guard 검출 곡선 — 임계값 2.0을 측정된 값으로 바꾸고, 어떤 통계량이
붕괴를 예측하는지 정한다.

**왜 이 실험인가.** 두 가지 구멍이 있다.

1. **임계값에 근거가 없다.** 도구는 여유 < 2.0이면 위험이라고 판정하는데 2.0의
   출처가 없다. 관측된 (여유, 붕괴) 쌍은 1.03→94.8%, 1.05→69.6%, 1.12→28.8%,
   1.15→20.4%, 2.85→0%, 2.90→0%뿐이고 **1.15와 2.85 사이가 통째로 비어 있다** —
   임계값을 놓은 바로 그 구간에 데이터가 하나도 없다.
2. **도구 검증이 통과하기 너무 쉽다.** 4개 모델 중 3개가 "안전"이라 무조건
   "안전"만 출력해도 3/4를 맞는다. 양성은 개발 대상 체크포인트 하나뿐이다.

이 스크립트는 `student_baseline_v2`의 마지막 블록 MLP를 여러 배수로 줄여
**여유 연속체**를 만들고, 각 지점에서 *도구가 보는 값*과 *실제 붕괴율*을 함께 잰다.
합격할 수도 불합격할 수도 있는 시험이 된다.

**세 번째 구멍도 같이 본다 — 통계량.** 도구는 여유를 이미지별 max의 **p50**으로
재는데, 붕괴는 **이미지 단위 이진 사건**이다. 10장 중 3장이 터져도 p50은 "안전"이라고
말할 수 있다. (6장의 계열 조사는 이미 p95를 함께 보고하고 있으므로, 이건 도구가
6장보다 후퇴한 것이다.) 여기서 p50·p95·최악을 모두 기록해 **어느 것이 붕괴를
예측하는지 데이터로 정한다.**

**사전 등록 (결과 보기 전 작성, 2026-09-15).**
  - 붕괴율이 어떤 여유에서 0으로 꺾이면 → 그 값에 마진을 붙여 임계값으로 채택하고
    2.0을 **측정된 값으로 교체**한다.
  - 꺾이는 점 없이 완만하면 → "단일 임계값으로 안전/위험을 가를 수 없다"를 결과로
    적고, 도구 출력을 이진 판정에서 **붕괴 확률 추정**으로 바꾼다.
  - 통계량은 최악(=가장 작은 여유)이 p50보다 나을 것으로 예상하나,
    **p50이 더 잘 맞으면 그대로 p50을 쓴다.**
  - 현재 임계값 2.0이 측정된 임계보다 **느슨하면**(즉 2.0에서도 붕괴가 있으면)
    그것은 도구의 결함이므로 그대로 보고하고 고친다.

실행:
    python scripts/detection_curve.py --limit 250
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images

BASE_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
FP16_MAX = 65504.0

# MLP 분기 축소 배수 f. 게이트형이므로 gate·up 각각 sqrt(f)를 곱한다(곱이 f배).
# 1.0(무개입)부터 촘촘히 내려 여유 0.9 → 4배 구간을 훑는다.
FACTORS = [1.0, 0.85, 0.72, 0.62, 0.55, 0.48, 0.42, 0.36, 0.30, 0.24, 0.18]


def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    i = min(int(q * (len(sorted_vals) - 1) + 0.5), len(sorted_vals) - 1)
    return sorted_vals[i]


def load_towers(checkpoint):
    """전체 3B를 **한 번만** fp32로 올려 비전 타워를 두 dtype으로 파생한다.

    dtype마다 from_pretrained를 부르면 4분짜리 로드를 두 번 한다. 또 bf16으로
    올린 뒤 fp16으로 변환하면 가수 비트가 7비트로 깎인 채 fp16이 되므로
    (fp32→fp16 직행은 10비트) 배포 조건과 달라진다. fp32에서 각각 파생한다.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_3B, torch_dtype=torch.float32, attn_implementation="eager")
    if checkpoint and checkpoint.lower() != "none":
        m = PeftModel.from_pretrained(m, checkpoint).merge_and_unload()
    vis = m.visual.eval()
    dep = copy.deepcopy(vis).to("cuda", torch.float16)
    mag = vis.to("cuda", torch.bfloat16)
    del m
    return mag, dep


def mlp_parts(tower):
    mlp = tower.blocks[-1].mlp
    return [mlp.gate_proj, mlp.up_proj]


def snapshot(tower):
    """원본 가중치·bias를 떠 둔다 — 배수마다 원본에서 다시 곱해야 누적되지 않는다."""
    return [(lin.weight.detach().clone(),
             None if lin.bias is None else lin.bias.detach().clone())
            for lin in mlp_parts(tower)]


@torch.no_grad()
def apply_factor(tower, snap, f):
    per_layer = math.sqrt(f)
    for lin, (w, b) in zip(mlp_parts(tower), snap):
        lin.weight.copy_(w).mul_(per_layer)
        if lin.bias is not None:
            lin.bias.copy_(b).mul_(per_layer)


@torch.no_grad()
def run_pass(tower, cache):
    """이미지별 (**blocks[-1] 출력**의 max|act|, 타워 출력의 유한 여부)를 돌려준다.

    cache는 **이미 GPU에 목표 dtype으로 올려 둔** (pixel_values, grid) 목록이다.
    CPU에 두고 매번 올리면 배수마다 34 MB × 장수를 PCIe로 넘기게 되어 전송이
    연산을 압도한다.

    **측정 대상 주의 (2026-09-15 실측으로 잡은 버그).** 처음에는 `tower(...)`의
    반환값에서 max를 쟀는데, 그건 merger를 통과한 뒤의 출력(absmax≈49)이라
    여유가 1,336배로 나오면서 같은 조건의 붕괴율 94.8%와 정면으로 모순됐다.
    문제의 활성은 merger **이전** 마지막 블록 출력(≈72,704)이다.
    8장의 도구가 `blocks[-1]`에 후크를 거는 것과 같은 지점을 봐야 한다.
    """
    rec = {}
    h = tower.blocks[-1].register_forward_hook(
        lambda _m, _i, o: rec.__setitem__(
            "t", (o[0] if isinstance(o, tuple) else o).detach()))
    maxes, finite = [], []
    try:
        for pv, grid in cache:
            out = tower(pv, grid)
            finite.append(bool(torch.isfinite(out).all()))
            blk = rec["t"].float()
            maxes.append(blk.abs().max().item() if torch.isfinite(blk).all()
                         else float("inf"))
    finally:
        h.remove()
    return maxes, finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/student_baseline_v2/epoch_1")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_pixels", type=int, default=0,
                    help="0이면 모델 기본값 (원본 해상도 경로)")
    ap.add_argument("--out", default="eval_results/detection_curve.json")
    ap.add_argument("--fig", default="figures/fig_detection_curve.png")
    ap.add_argument("--factors", default=None,
                    help="쉼표로 구분한 배수 목록. 전이 구간을 촘촘히 다시 훑을 때 쓴다")
    args = ap.parse_args()

    global FACTORS
    if args.factors:
        FACTORS = [float(x) for x in args.factors.split(",")]
        print(f"[배수] 사용자 지정 {len(FACTORS)}개: {FACTORS}")

    awq_compat.patch()
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE_3B)
    if args.max_pixels:
        proc.image_processor.max_pixels = args.max_pixels
    mp = proc.image_processor.max_pixels
    print(f"[설정] max_pixels={mp:,}  체크포인트={args.checkpoint}")

    tok = build_token_to_images(DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.limit]
    print(f"[이미지] {len(paths)}장 전처리 중...", flush=True)

    # 두 dtype 사본을 **미리 GPU에 올려 둔다** (250장 × 7296 × 1176 × 2바이트 ≈ 4.3 GB씩).
    cache_mag, cache_dep = [], []
    for p in paths:
        enc = proc(text=["<image>"], images=[Image.open(p).convert("RGB")],
                   return_tensors="pt")
        pv, grid = enc["pixel_values"], enc["image_grid_thw"].to("cuda")
        cache_mag.append((pv.to("cuda", torch.bfloat16), grid))
        cache_dep.append((pv.to("cuda", torch.float16), grid))
    gib = sum(t.numel() * t.element_size() for t, _ in cache_mag + cache_dep) / 2**30
    print(f"[이미지] 완료. pixel_values={tuple(cache_mag[0][0].shape)}  "
          f"GPU 상주 {gib:.1f} GiB", flush=True)

    # 두 타워를 따로 둔다. bf16 타워는 **크기의 참값**을, fp16 타워는 **실제 붕괴**를
    # 잰다 (8장의 설계 판단). 각각 자기 dtype으로 반올림된 가중치를 쓰므로
    # 실제 배포와 같은 조건이다.
    print("[모델] fp32로 1회 로드 후 bf16/fp16 타워 파생...", flush=True)
    tw_mag, tw_dep = load_towers(args.checkpoint)
    snap_mag, snap_dep = snapshot(tw_mag), snapshot(tw_dep)
    print("[모델] 준비 완료", flush=True)

    rows = []
    for f in FACTORS:
        apply_factor(tw_mag, snap_mag, f)
        apply_factor(tw_dep, snap_dep, f)

        maxes, _ = run_pass(tw_mag, cache_mag)
        _, finite = run_pass(tw_dep, cache_dep)

        sm = sorted(maxes)
        h50 = FP16_MAX / pct(sm, 0.50)
        h95 = FP16_MAX / pct(sm, 0.95)      # 큰 활성 쪽 = 작은 여유
        hworst = FP16_MAX / sm[-1]
        collapse = 1.0 - sum(finite) / len(finite)
        rows.append({"factor": f, "headroom_p50": h50, "headroom_p95": h95,
                     "headroom_worst": hworst, "collapse_rate": collapse,
                     "n_images": len(finite),
                     "verdict_at_2.0": "위험" if h50 < 2.0 else "안전",
                     "max_p50": pct(sm, 0.50), "max_worst": sm[-1],
                     # **이미지별 값을 남긴다 (2026-09-15 추가).** 게이트를 p50에서
                     # 최악으로 바꾸면 "몇 장을 봐야 그 최악이 믿을 만한가"가 곧바로
                     # 따라온다. 이미지별 max를 저장해 두면 worst-of-N의 수렴을
                     # **부트스트랩으로** 답할 수 있어 재실행이 필요 없다.
                     "per_image_max": maxes,
                     "per_image_finite": finite})
        print(f"  f={f:<5.2f}  여유 p50={h50:6.2f}  p95={h95:6.2f}  최악={hworst:6.2f}  "
              f"붕괴={collapse*100:5.1f}%", flush=True)

    # ---- 분석 ----
    print("\n" + "=" * 92)
    print(f"{'f':>6}{'여유 p50':>11}{'여유 p95':>11}{'여유 최악':>11}"
          f"{'붕괴율':>10}{'2.0 판정':>10}{'실제':>8}{'일치':>7}")
    print("-" * 92)
    conf = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for r in rows:
        truth = "위험" if r["collapse_rate"] > 0 else "안전"
        pred = r["verdict_at_2.0"]
        ok = "○" if pred == truth else "**×**"
        conf["TP" if (pred == "위험" and truth == "위험") else
             "FP" if (pred == "위험") else
             "FN" if (truth == "위험") else "TN"] += 1
        print(f"{r['factor']:>6.2f}{r['headroom_p50']:>11.2f}{r['headroom_p95']:>11.2f}"
              f"{r['headroom_worst']:>11.2f}{r['collapse_rate']*100:>9.1f}%"
              f"{pred:>10}{truth:>8}{ok:>7}")
    print("=" * 92)

    # 붕괴가 0이 되는 최초 지점 (f가 작아지는 순서)
    safe = [r for r in rows if r["collapse_rate"] == 0.0]
    risky = [r for r in rows if r["collapse_rate"] > 0.0]
    summary = {}
    if safe and risky:
        last_risky = max(risky, key=lambda r: r["headroom_p50"])
        first_safe = min(safe, key=lambda r: r["headroom_p50"])
        summary = {
            "last_collapsing": {k: last_risky[k] for k in
                                ("factor", "headroom_p50", "headroom_p95",
                                 "headroom_worst", "collapse_rate")},
            "first_safe": {k: first_safe[k] for k in
                           ("factor", "headroom_p50", "headroom_p95",
                            "headroom_worst", "collapse_rate")},
        }
        print(f"\n[전이 구간] 붕괴 마지막 지점 여유 p50={last_risky['headroom_p50']:.2f} "
              f"(붕괴 {last_risky['collapse_rate']*100:.1f}%)  →  "
              f"최초 무붕괴 여유 p50={first_safe['headroom_p50']:.2f}")
        print(f"             최악 기준으로는 {last_risky['headroom_worst']:.2f} → "
              f"{first_safe['headroom_worst']:.2f}")

    # 어느 통계량이 붕괴를 더 잘 가르는가 — 분리 여백(안전 최소 − 위험 최대)
    print("\n[통계량 비교] 안전/위험을 가르는 여백 (클수록 좋은 예측자)")
    best = None
    for key in ("headroom_p50", "headroom_p95", "headroom_worst"):
        if not (safe and risky):
            break
        lo_safe = min(r[key] for r in safe)
        hi_risky = max(r[key] for r in risky)
        margin = lo_safe - hi_risky
        sep = "분리됨" if margin > 0 else "**겹침**"
        print(f"  {key:<18} 안전 최소={lo_safe:6.2f}  위험 최대={hi_risky:6.2f}  "
              f"여백={margin:+6.2f}  {sep}")
        if best is None or margin > best[1]:
            best = (key, margin, lo_safe, hi_risky)
    if best:
        summary["best_statistic"] = {"name": best[0], "margin": best[1],
                                     "safe_min": best[2], "risky_max": best[3]}
        print(f"\n  → 가장 잘 가르는 통계량: **{best[0]}** (여백 {best[1]:+.2f})")

    print("\n[현재 임계값 2.0 성적]  "
          f"TP={conf['TP']} TN={conf['TN']} FP={conf['FP']} **FN={conf['FN']}**")
    if conf["FN"]:
        print("  **미탐이 있다 — 2.0은 느슨하다. 도구의 결함이므로 임계값을 올린다.**")
    elif conf["FP"]:
        print(f"  오탐 {conf['FP']}건 — 안전한데 위험이라 한다. 보수적이지만 "
              "불필요한 가중치 편집을 유발하므로 비용이다.")
    else:
        print("  이 구간에서 2.0은 오탐·미탐 없이 동작한다.")

    out = {"checkpoint": args.checkpoint, "max_pixels": mp,
           "n_images": len(cache_mag), "factors": FACTORS,
           "rows": rows, "confusion_at_2.0": conf, "summary": summary}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n저장: {args.out}")

    # ---- 그림 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for key, style, lab in (("headroom_p50", "o-", "headroom (p50)"),
                                ("headroom_worst", "s--", "headroom (worst image)")):
            xs = [r[key] for r in rows]
            ys = [r["collapse_rate"] * 100 for r in rows]
            o = sorted(zip(xs, ys))
            ax.plot([a for a, _ in o], [b for _, b in o], style, ms=4, label=lab)
        ax.axvline(2.0, color="crimson", ls=":", lw=1.4, label="current threshold 2.0")
        ax.set_xlabel("Representable-range headroom  (65,504 / max|activation|)")
        ax.set_ylabel("Collapse rate (%)")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        Path(args.fig).parent.mkdir(exist_ok=True)
        fig.savefig(args.fig, dpi=180)
        fig.savefig(args.fig.replace(".png", ".pdf"))
        print(f"그림: {args.fig}")
    except Exception as e:
        print(f"그림 생략: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
