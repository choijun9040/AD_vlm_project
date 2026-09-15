"""두 조건의 문항별 정오를 짝지어 McNemar 검정한다 (fixA 비용 확정용).

**왜 필요한가.** fixA의 -0.59%p는 11,309문항 중 67문항이다. 우리가 "구별 안 됨"이라
부르는 5-way의 0.35%p는 40문항이다. 1.7배 차이일 뿐이므로, 한쪽은 "잡음"이라 하고
다른 쪽은 "비용"이라 하려면 **같은 기준으로 재야 한다.**

fixA 비교는 5-way와 달리 **학습 무작위성이 없다**(같은 가중치에 결정론적 편집).
따라서 남는 잡음원은 평가 표본뿐이고, 그것이 바로 McNemar가 검정하는 것이다.

실행:
    python scripts/analyze_mcnemar_pair.py \
        --a eval_results/posthoc_none_predictions.jsonl \
        --b eval_results/posthoc_fixa_predictions.jsonl
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def chi2_sf_df1(x):
    return math.erfc(math.sqrt(x / 2.0)) if x > 0 else 1.0


def load(f):
    out = {}
    for line in Path(f).open():
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["idx"]] = (bool(r["correct"]), r.get("template_type"))
    return out


def mcnemar(a, b, keys):
    n_a = sum(1 for k in keys if a[k][0] and not b[k][0])   # A만 맞힘
    n_b = sum(1 for k in keys if b[k][0] and not a[k][0])   # B만 맞힘
    n = n_a + n_b
    if n == 0:
        return n_a, n_b, 0.0, 1.0
    stat = (abs(n_a - n_b) - 1) ** 2 / n
    return n_a, n_b, stat, chi2_sf_df1(stat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 조건 JSONL (예: 무개입)")
    ap.add_argument("--b", required=True, help="비교 조건 JSONL (예: fixA)")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--by_template", action="store_true")
    ap.add_argument("--out", default="eval_results/mcnemar_pair.json")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    keys = sorted(set(A) & set(B))
    acc_a = sum(A[k][0] for k in keys) / len(keys)
    acc_b = sum(B[k][0] for k in keys) / len(keys)
    n_a, n_b, stat, p = mcnemar(A, B, keys)

    print(f"공통 문항 {len(keys):,}개\n")
    print(f"  {args.label_a:<12} {acc_a*100:.2f}%")
    print(f"  {args.label_b:<12} {acc_b*100:.2f}%   차이 {(acc_b-acc_a)*100:+.2f}pp "
          f"({round((acc_b-acc_a)*len(keys)):+d}문항)")
    print(f"\n  불일치 쌍 {n_a+n_b:,}개  ({args.label_a}만 맞힘 {n_a:,} / "
          f"{args.label_b}만 맞힘 {n_b:,})")
    print(f"  McNemar χ²(1) = {stat:.2f},  p = {p:.4g}")
    sig = p < 0.05
    print(f"\n  판정: {'**유의** — 평가 표본 잡음으로 설명되지 않는다' if sig else '유의하지 않음 — 평가 표본 잡음과 구별되지 않는다'}")
    if sig:
        print(f"    → 차이를 실제 비용으로 서술할 수 있다. 다만 이는 **이 두 조건**에")
        print(f"      대한 진술이며, 학습 무작위성이 없는 결정론적 개입이라 성립한다.")
    else:
        print(f"    → '비용 {abs(acc_b-acc_a)*100:.2f}pp'라고 단정하지 말 것.")
        print(f"      '측정 한계 이하'로 서술하는 것이 정확하다.")

    res = {"a": args.a, "b": args.b, "n_items": len(keys),
           "acc_a": acc_a, "acc_b": acc_b, "diff_pp": (acc_b - acc_a) * 100,
           "a_only": n_a, "b_only": n_b, "n_discordant": n_a + n_b,
           "chi2": stat, "p": p, "significant": bool(sig)}

    if args.by_template:
        print(f"\n{'유형':<14}{'n':>7}{'차이(pp)':>10}{'불일치':>8}{'p':>9}")
        tt = Counter(A[k][1] for k in keys)
        for t in sorted(tt):
            sub = [k for k in keys if A[k][1] == t]
            aa = sum(A[k][0] for k in sub) / len(sub)
            bb = sum(B[k][0] for k in sub) / len(sub)
            x, y, st, pp = mcnemar(A, B, sub)
            print(f"{str(t):<14}{len(sub):>7}{(bb-aa)*100:>+10.2f}{x+y:>8}{pp:>9.4f}")
            res.setdefault("by_template", {})[str(t)] = {
                "n": len(sub), "diff_pp": (bb - aa) * 100,
                "n_discordant": x + y, "p": pp}

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
