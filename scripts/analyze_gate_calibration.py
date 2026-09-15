"""`headroom_guard` 게이트 보정 — 임계값·통계량·표본 수를 데이터로 정한다.

**세 가지를 한꺼번에 답한다.**
  1. **통계량**: 여유를 p50으로 볼 것인가, p95로 볼 것인가, 최악으로 볼 것인가.
     붕괴는 이미지 단위 이진 사건이므로 중앙값은 원리적으로 맞지 않지만,
     "맞지 않는다"와 "실제로 틀린다"는 다르므로 실측으로 가른다.
  2. **임계값**: 현재 2.0은 출처가 없는 매직 넘버다. 안전/위험이 갈리는 지점을
     검출 곡선(합성)과 공개 모델(자연)에서 함께 읽는다.
  3. **표본 수**: 게이트를 최악으로 바꾸면 "몇 장의 최악이냐"가 곧바로 따라온다.
     최악은 N에 따라 편향된 추정량이다. 이미지별 값에서 **부트스트랩**으로
     worst-of-N의 분포를 그려, N을 줄였을 때 판정이 얼마나 낙관적으로
     기우는지 정량화한다.

**입력** `detection_curve.json`(합성 연속체) · `detection_curve_fine.json`(전이 정밀)
        · `natural_collapse.json`(공개 모델의 실제 붕괴율)
**출력** 권고 규칙 `(통계량, 임계값, 최소 표본 수)`과 그 규칙의 혼동 행렬

**사전 등록 (결과 보기 전, 2026-09-15).**
  - 세 통계량 중 안전/위험을 **여백 있게 분리**하는 것을 채택한다. 동률이면
    보수적인 쪽(최악)을 쓴다.
  - 임계값은 "붕괴가 관측된 최대 여유"와 "붕괴가 없는 최소 여유" 사이에서 잡되,
    **관측된 경계에 딱 붙이지 않는다** — 표본 밖 이미지를 위한 여유를 남긴다.
    그 여유의 크기는 부트스트랩이 말하는 worst-of-N 편향에서 가져온다.
  - 현행 2.0이 과보수로 판명되어도 **1.0 근처까지 내리지는 않는다.** 오탐은
    불필요한 가중치 편집(등가가 아님)을 부르고 미탐은 배포 실패를 부르므로
    두 오류의 비용이 대칭이 아니다. 이 비대칭을 명시하고 결정한다.

실행:
    python scripts/analyze_gate_calibration.py
"""
import argparse
import json
import random
from pathlib import Path

FP16_MAX = 65504.0
STATS = ("headroom_p50", "headroom_p95", "headroom_worst")
LABEL = {"headroom_p50": "p50", "headroom_p95": "p95", "headroom_worst": "최악"}


def load_rows(paths):
    rows = []
    for p in paths:
        f = Path(p)
        if not f.exists():
            print(f"  (없음, 건너뜀: {p})")
            continue
        d = json.loads(f.read_text())
        for r in d.get("rows", []):
            r["source"] = f.stem
            rows.append(r)
    return rows


def natural_rows(path):
    f = Path(path)
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    out = []
    for tag, r in d.items():
        if "collapse_rate" not in r:
            continue
        out.append({"source": "natural", "tag": tag,
                    "headroom_p50": r.get("headroom_p50_bf16"),
                    "headroom_p95": r.get("headroom_p95_bf16"),
                    "headroom_worst": None,      # 계열 조사는 최악을 남기지 않았다
                    "collapse_rate": r["collapse_rate"]})
    return out


def separation(rows, key):
    """안전(붕괴 0)과 위험(붕괴 > 0)을 이 통계량이 얼마나 갈라놓는가."""
    safe = [r[key] for r in rows if r.get(key) is not None and r["collapse_rate"] == 0]
    risky = [r[key] for r in rows if r.get(key) is not None and r["collapse_rate"] > 0]
    if not safe or not risky:
        return None
    return {"safe_min": min(safe), "risky_max": max(risky),
            "margin": min(safe) - max(risky), "n_safe": len(safe), "n_risky": len(risky)}


def bootstrap_worst(per_image_max, sizes, trials=2000, seed=0):
    """N장을 복원추출했을 때 worst-of-N 여유의 분포. 100장 최악과 비교한다."""
    rnd = random.Random(seed)
    vals = [v for v in per_image_max if v == v and v != float("inf")]
    if len(vals) < 10:
        return None
    full_worst = FP16_MAX / max(vals)
    out = {"n_images": len(vals), "worst_full": full_worst, "by_n": {}}
    for n in sizes:
        if n > len(vals):
            continue
        est = []
        for _ in range(trials):
            est.append(FP16_MAX / max(rnd.choice(vals) for _ in range(n)))
        est.sort()
        # 낙관 편향: 표본 최악이 전체 최악보다 '여유가 크다'고 말하는 정도
        out["by_n"][n] = {
            "median": est[len(est) // 2],
            "p10": est[int(0.10 * len(est))],
            "p90": est[int(0.90 * len(est))],
            "optimism_median": est[len(est) // 2] / full_worst,
            "p_overestimate_20pct": sum(1 for e in est if e > full_worst * 1.2) / len(est),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", nargs="*", default=[
        "eval_results/detection_curve.json", "eval_results/detection_curve_fine.json"])
    ap.add_argument("--natural", default="eval_results/natural_collapse.json")
    ap.add_argument("--out", default="eval_results/gate_calibration.json")
    args = ap.parse_args()

    syn = load_rows(args.curves)
    nat = natural_rows(args.natural)
    allr = syn + nat
    print(f"합성 점 {len(syn)}개 · 자연 점 {len(nat)}개\n")

    # ---- 1. 통계량 ----
    print("[1] 통계량 — 안전/위험 분리 여백 (클수록 좋은 예측자)")
    print(f"  {'통계량':<8}{'위험 최대':>10}{'안전 최소':>10}{'여백':>9}  판정")
    seps, best = {}, None
    for k in STATS:
        s = separation(allr, k)
        seps[k] = s
        if s is None:
            print(f"  {LABEL[k]:<8}{'—':>10}{'—':>10}{'—':>9}  (한쪽 없음)"); continue
        verdict = "분리됨" if s["margin"] > 0 else "**겹침 — 이 통계량으로는 못 가른다**"
        print(f"  {LABEL[k]:<8}{s['risky_max']:>10.2f}{s['safe_min']:>10.2f}"
              f"{s['margin']:>+9.2f}  {verdict}")
        if s["margin"] > 0 and (best is None or s["margin"] > seps[best]["margin"]):
            best = k
    print(f"\n  → 채택: **{LABEL[best] if best else '없음'}**")

    # ---- 2. 임계값 ----
    result = {"separation": seps, "best_statistic": best}
    if best:
        s = seps[best]
        print(f"\n[2] 임계값 — {LABEL[best]} 기준")
        print(f"  붕괴가 관측된 최대 여유 : {s['risky_max']:.3f}")
        print(f"  붕괴가 없는 최소 여유   : {s['safe_min']:.3f}")
        print(f"  현행 임계값 2.0은 이 경계보다 {2.0 / s['safe_min']:.2f}배 보수적")
        result["boundary"] = {"risky_max": s["risky_max"], "safe_min": s["safe_min"]}

    # ---- 3. 표본 수 ----
    print("\n[3] 표본 수 — worst-of-N 부트스트랩 (전이 부근 배수들)")
    boots = {}
    cand = [r for r in syn if "per_image_max" in r]
    if not cand:
        print("  이미지별 값이 저장된 실행이 없다 — detection_curve_fine 완료 후 다시 볼 것")
    for r in sorted(cand, key=lambda x: x["headroom_worst"])[:4]:
        b = bootstrap_worst(r["per_image_max"], [5, 10, 20, 30, 50, 100])
        if b is None:
            continue
        boots[f"f={r['factor']}"] = b
        print(f"\n  배수 f={r['factor']}  (100장 최악 여유 {b['worst_full']:.3f}, "
              f"붕괴 {r['collapse_rate']*100:.0f}%)")
        print(f"    {'N':>5}{'최악 중앙값':>13}{'p10':>9}{'p90':>9}"
              f"{'낙관 배율':>10}{'20%↑ 과대 확률':>15}")
        for n, v in b["by_n"].items():
            print(f"    {n:>5}{v['median']:>13.3f}{v['p10']:>9.3f}{v['p90']:>9.3f}"
                  f"{v['optimism_median']:>10.2f}배{v['p_overestimate_20pct']*100:>14.1f}%")
    result["bootstrap"] = boots

    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
