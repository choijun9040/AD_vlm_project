"""
장면 군집 부트스트랩 신뢰구간 — 평가 데이터 전부로 통일 (사전 등록 4fddeb4).

799장은 149장면에 묶여 있어 이미지가 서로 독립이 아니다. 그래서 **장면을 복원 추출**해 신뢰구간을 낸다.
규칙(사전 등록): 10,000회, 백분위 95% 구간, 시드 1234. 0건 관측의 상한은 장면 수 기준 이항 상한
1 − 0.05^(1/장면 수)로 적는다(이미지 수 기준이 아니다).

    # 비율 — 필드가 참인 이미지의 비율
    python scripts/cluster_bootstrap.py rate  --file X.jsonl --field nonfinite_float16 --where phase=before
    # 임계 초과 비율 — 값이 65504를 넘는 이미지의 비율
    python scripts/cluster_bootstrap.py exceed --file X.jsonl --field blk_last_max --threshold 65504
    # 평균·중앙값
    python scripts/cluster_bootstrap.py mean  --file X.jsonl --field blk_last_max --stat median
    # 짝지은 차이 — 같은 이미지(키)끼리 A − B
    python scripts/cluster_bootstrap.py paired --file A.jsonl --file_b B.jsonl --field correct --key token
"""
import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict

N_BOOT = 10_000
SEED = 1234


def load(path, where):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for cond in where or []:
        k, v = cond.split("=", 1)
        rows = [r for r in rows if str(r.get(k)) == v]
    return rows


def by_scene(rows, value, group="scene"):
    g = defaultdict(list)
    for r in rows:
        if r.get(group) is None:
            raise SystemExit(f"장면 토큰이 없는 행이 있다: {r.get('image') or r}")
        g[r[group]].append(value(r))
    return g


def boot(groups, stat, n=N_BOOT, seed=SEED):
    rng = random.Random(seed)
    keys = list(groups)
    est = []
    for _ in range(n):
        vals = []
        for k in (keys[rng.randrange(len(keys))] for _ in keys):
            vals.extend(groups[k])
        est.append(stat(vals))
    est.sort()
    return est[int(0.025 * n)], est[int(0.975 * n) - 1]


def zero_bound(n_scene, conf=0.95):
    return 1 - (1 - conf) ** (1 / n_scene)


def report(name, point, lo, hi, n_img, n_scene, zero=False):
    out = {"stat": name, "point": point, "ci95": [lo, hi], "n_images": n_img, "n_scenes": n_scene,
           "n_boot": N_BOOT, "seed": SEED}
    if zero:
        out["zero_upper_bound_by_scene"] = zero_bound(n_scene)
    print(json.dumps(out, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", choices=["rate", "exceed", "mean", "paired"])
    ap.add_argument("--file", required=True)
    ap.add_argument("--file_b", help="paired의 B 파일 (없으면 --where_b로 같은 파일에서 고른다)")
    ap.add_argument("--field", required=True)
    ap.add_argument("--field_b", help="paired에서 B 쪽 필드 (기본은 --field)")
    ap.add_argument("--where", nargs="*", help="key=value 필터")
    ap.add_argument("--where_b", nargs="*")
    ap.add_argument("--threshold", type=float, default=65504.0)
    ap.add_argument("--stat", choices=["mean", "median"], default="mean")
    ap.add_argument("--key", default="image", help="paired에서 짝을 맞출 키")
    ap.add_argument("--group", default="scene")
    args = ap.parse_args()

    rows = load(args.file, args.where)
    if args.cmd in ("rate", "exceed"):
        if args.cmd == "rate":
            val = lambda r: 1.0 if r[args.field] else 0.0
        else:
            val = lambda r: 1.0 if (r[args.field] is None or not math.isfinite(r[args.field])
                                    or r[args.field] > args.threshold) else 0.0
        g = by_scene(rows, val, args.group)
        allv = [v for vs in g.values() for v in vs]
        point = sum(allv) / len(allv)
        lo, hi = boot(g, lambda v: sum(v) / len(v))
        report(f"{args.cmd}:{args.field}", point, lo, hi, len(allv), len(g), zero=(point == 0))
    elif args.cmd == "mean":
        f = st.median if args.stat == "median" else st.mean
        g = by_scene([r for r in rows if r[args.field] is not None and math.isfinite(r[args.field])],
                     lambda r: r[args.field], args.group)
        allv = [v for vs in g.values() for v in vs]
        lo, hi = boot(g, f)
        report(f"{args.stat}:{args.field}", f(allv), lo, hi, len(allv), len(g))
    else:
        rows_b = load(args.file_b or args.file, args.where_b)
        fb = args.field_b or args.field
        b = {r[args.key]: r for r in rows_b}
        num = lambda x: float(x) if not isinstance(x, bool) else (1.0 if x else 0.0)
        pairs = [dict(r, _d=num(r[args.field]) - num(b[r[args.key]][fb])) for r in rows if r[args.key] in b]
        g = by_scene(pairs, lambda r: r["_d"], args.group)
        allv = [v for vs in g.values() for v in vs]
        lo, hi = boot(g, lambda v: sum(v) / len(v))
        report(f"paired_mean_diff:{args.field}-{fb}", sum(allv) / len(allv), lo, hi, len(allv), len(g))


if __name__ == "__main__":
    main()
