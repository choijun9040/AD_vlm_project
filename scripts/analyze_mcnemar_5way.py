"""
5-way 정확도 차이가 **평가 표본 잡음과 구별되는가** — 문항 단위 McNemar 검정.

**왜 필요한가.** 5개 변형은 시드 고정 없이 한 번씩만 학습했으므로 전체 정확도
0.35%p 차이(11,309문항 중 40문항)를 순위로 읽을 수 없다. 지금까지 논문은
"순서를 주장하지 않는다"를 **관행적 조심**으로 써 왔는데, 이 검정을 하면
**검정 결과**가 된다.

**시드 분산을 재지 않고도 결론이 나는 이유.** McNemar는 같은 문항에 대한 두 모델의
정오를 짝지어, **평가 표본 잡음만으로** 그 차이가 설명되는지 본다. 시드 분산이 0이라
가정한 가장 유리한 조건인데도 유의하지 않다면, 시드 분산을 더하면 불확실성은 더
커질 뿐이므로 결론은 그대로다. (반대로 유의하게 나오면 시드 분산을 재야 한다.)

두 모델이 **다르게 답한 불일치 쌍**만이 정보를 갖는다. b = A만 맞힌 수,
c = B만 맞힌 수일 때 검정통계량은 (|b-c|-1)^2/(b+c) (연속성 보정, 자유도 1).

실행:
    python scripts/analyze_mcnemar_5way.py
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

VARIANTS = [
    ("full", "student_full_epoch_1"),
    ("spatial", "student_spatial_epoch_1"),
    ("temporal", "student_temporal_epoch_1"),
    ("baseline_v2", "student_baseline_v2_epoch_1"),
    ("kd_only_v4", "student_kd_only_v4_epoch_1"),
]


def chi2_sf_df1(x):
    """자유도 1 카이제곱 상측 확률 = erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(x / 2.0)) if x > 0 else 1.0


def load(tag, d):
    p = Path(d) / f"{tag}_predictions.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[r["idx"]] = (bool(r["correct"]), r.get("template_type"))
    return out


def mcnemar(a, b, keys):
    b_only = sum(1 for k in keys if a[k][0] and not b[k][0])
    c_only = sum(1 for k in keys if b[k][0] and not a[k][0])
    n_dis = b_only + c_only
    if n_dis == 0:
        return b_only, c_only, 0.0, 1.0
    stat = (abs(b_only - c_only) - 1) ** 2 / n_dis
    return b_only, c_only, stat, chi2_sf_df1(stat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval_results")
    ap.add_argument("--template", default=None,
                    help="특정 template_type만 (예: object)")
    ap.add_argument("--out", default="eval_results/mcnemar_5way.json")
    args = ap.parse_args()

    data, missing = {}, []
    for name, tag in VARIANTS:
        d = load(tag, args.dir)
        if d is None:
            missing.append(tag)
        else:
            data[name] = d
    if missing:
        print("덤프 없음 — 먼저 --dump-predictions로 평가할 것:")
        for m in missing:
            print(f"  {m}_predictions.jsonl")
        if not data:
            raise SystemExit(1)

    keys = set.intersection(*(set(v) for v in data.values()))
    if args.template:
        keys = {k for k in keys
                if next(iter(data.values()))[k][1] == args.template}
    keys = sorted(keys)
    label = args.template or "전체"
    print(f"[대상] {label}, 공통 문항 {len(keys):,}개, 변형 {len(data)}개\n")

    acc = {n: sum(1 for k in keys if data[n][k][0]) / len(keys) for n in data}
    print(f"{'변형':<14}{'정확도':>9}")
    for n, a in sorted(acc.items(), key=lambda x: -x[1]):
        print(f"{n:<14}{a*100:>8.2f}%")

    print(f"\n{'쌍':<28}{'차이(pp)':>10}{'불일치':>8}{'b':>7}{'c':>7}{'χ²':>8}{'p':>9}  판정")
    rows = []
    for x, y in combinations(data, 2):
        b, c, stat, p = mcnemar(data[x], data[y], keys)
        diff = (acc[x] - acc[y]) * 100
        sig = "유의" if p < 0.05 else "구별 안 됨"
        print(f"{x+' vs '+y:<28}{diff:>+10.2f}{b+c:>8}{b:>7}{c:>7}{stat:>8.2f}{p:>9.4f}  {sig}")
        rows.append({"a": x, "b": y, "diff_pp": diff, "n_discordant": b + c,
                     "a_only": b, "b_only": c, "chi2": stat, "p": p,
                     "significant": bool(p < 0.05)})

    print()
    key3 = [r for r in rows if {r["a"], r["b"]} <= {"full", "spatial", "temporal"}]
    if key3:
        ns = [r for r in key3 if not r["significant"]]
        print(f"핵심 3변형(full/spatial/temporal) 쌍 {len(key3)}개 중 "
              f"**구별 안 됨 {len(ns)}개**")
        if len(ns) == len(key3):
            print("  → 시드 분산을 가정하지 않고 **평가 표본 잡음만으로도** 세 변형이")
            print("     구별되지 않는다. '순서를 주장하지 않는다'가 검정으로 뒷받침된다.")
        else:
            print("  → 일부 쌍이 유의하다. 평가 잡음만으로는 설명되지 않는다.")
            print("     그러나 **방법이 다르다는 뜻은 아니다**(아래 참조). 서술을 재검토할 것.")

    print()
    print("  [해석 주의] 이 검정의 귀무가설은 '이 **두 학습된 체크포인트**가 같은")
    print("  정확도를 갖는다'이지 '이 두 **학습 방법**이 같다'가 아니다. 각 방법에")
    print("  체크포인트가 하나뿐이므로 **방법 수준 순위 주장은 이 검정 결과와 무관하게**")
    print("  뒷받침되지 않는다(시드 반복이 필요하며 5변형 × 3시드 ≈ 32일이라 생략했다).")
    print("  유의하지 않게 나온 경우: 체크포인트끼리도 구별되지 않으므로 방법 수준")
    print("    질문이 무의미해진다 — 논문이 필요로 하는 결론은 이것으로 충분하다.")
    print("  유의하게 나온 경우: '이 두 체크포인트는 다르다'까지만 말할 수 있고,")
    print("    방법 차이인지 시드 차이인지는 **여전히 알 수 없다.**")
    print("  어느 경우에도 '차이가 없음이 증명됐다'고 쓰지 말 것 — 비유의는 동등성의")
    print("    증거가 아니다.")

    Path(args.out).write_text(json.dumps(
        {"template": label, "n_items": len(keys), "accuracy": acc, "pairs": rows},
        ensure_ascii=False, indent=2))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
