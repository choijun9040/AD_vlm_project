"""문서에 인용된 수치들이 **같은 조건에서 왔는지** 검사한다.

**왜 이 검사가 따로 필요한가.** 값 대조만으로는 못 잡는 오류가 있다.
2026-09-15 하루에 세 건이 나왔고 셋 다 각 숫자는 자기 파일과 정확히 일치했다.

  1. Table 1의 여유 열이 **fp16에서 잰 크기**로 계산돼 있었다. 붕괴율 열은 bf16
     실행에서 왔으므로 한 표 안에서 측정 dtype이 갈렸고, 그 결과
     "여유 1.03배(= 한계 미초과)인데 94.8% 붕괴"라는 자기모순이 생겼다.
  2. Table 1의 INT8 열이 **평가 해상도**(200,704)에서, 여유·붕괴 열은 **원본
     해상도**(1,440,000)에서 온 값이었다. 표 제목은 "at native resolution"이었다.
  3. λ 용량-반응 수치를 본문은 250장 실행에서, Fig. 2는 799장 실행에서 가져왔다.
     독자가 문장과 그림을 대조하면 끝점이 어긋난다.

세 건 모두 **값 대조(25/25 통과)로는 잡히지 않는다.** 잡으려면 함께 인용된 숫자들의
`_meta`(해상도·표본 수·dtype)를 서로 비교해야 한다. 이 스크립트가 그 일을 한다.

**쓰는 법.** `docs/claim_provenance.yaml`에 "한 표/한 문단에서 함께 읽히는 숫자들"을
그룹으로 묶고, 그 그룹이 참조하는 **모든** 산출물(그림의 원본 데이터 포함)을 적는다.
스크립트는 그룹 안의 메타데이터가 갈리면 보고한다. 갈려도 되는 항목은 `allow_differ`에
**이유와 함께** 적어야 통과한다 — 이유를 쓰게 만드는 것이 이 도구의 요점이다.

값 대조도 함께 한다(`values:`). 경로는 `파일::a.b.c` 형식이고 `expr`로 변환을 준다
(`x`가 추출값, 예: `65504/x`, `100*(1-x/250)`).

**검사기 자신의 회귀 시험.** 항상 통과하는 검사는 쓸모가 없다.
`docs/claim_provenance_regress.yaml`은 위 결함 ②③을 재현해 둔 매니페스트이며
**반드시 불합격**해야 한다(종료코드 1). 검사기를 고칠 때마다 먼저 돌려 볼 것.

실행:
    python scripts/verify_claim_provenance.py
    python scripts/verify_claim_provenance.py --manifest docs/claim_provenance.yaml
    python scripts/verify_claim_provenance.py \
        --manifest docs/claim_provenance_regress.yaml --strict   # 종료코드 1이어야 정상
"""
import argparse
import json
import math
import sys
from pathlib import Path

# 메타로 취급할 키. `_meta` 안에 있든 최상위에 있든 찾는다.
META_KEYS = ("max_pixels", "min_pixels", "n_images", "dtype", "limit",
             "n_checked", "target_headroom", "trt", "checkpoint", "modes")


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"  [경고] {path} 파싱 실패: {e}")
        return None


def extract_meta(doc):
    """`_meta`가 있으면 그것을, 없으면 최상위에서 메타 키를 긁는다."""
    if not isinstance(doc, dict):
        return {}
    meta = {}
    src = doc.get("_meta") if isinstance(doc.get("_meta"), dict) else doc
    for k in META_KEYS:
        if k in src and not isinstance(src[k], (dict, list)):
            meta[k] = src[k]
    # `_meta`가 따로 있어도 최상위에 조건이 적힌 산출물이 있다(guard 계열)
    if doc.get("_meta"):
        for k in META_KEYS:
            if k in doc and k not in meta and not isinstance(doc[k], (dict, list)):
                meta[k] = doc[k]
    return meta


def dig(doc, path):
    """점으로 구분된 경로를 따라간다.

    **키 자체에 점이 들어 있는 경우를 처리한다** — `sweep_align0.0`,
    `AngelSlim/Qwen2.5-VL-3B-Instruct-INT4-AWQ` 같은 키가 실제로 있다.
    단순 split(".")은 여기서 깨지므로, 각 단계에서 **남은 경로의 접두사와
    일치하는 가장 긴 키**를 고른다.
    """
    cur, rest = doc, path
    while rest:
        if isinstance(cur, list):
            seg, _, rest = rest.partition(".")
            cur = cur[int(seg)]
            continue
        if not isinstance(cur, dict):
            raise KeyError(f"{rest} — 더 내려갈 수 없다 ({type(cur).__name__})")
        cand = [k for k in cur if rest == k or rest.startswith(k + ".")]
        if not cand:
            raise KeyError(f"{rest} (가능한 키: {list(cur)[:6]})")
        k = max(cand, key=len)
        cur = cur[k]
        rest = rest[len(k):].lstrip(".")
    return cur


def check_group(g, strict):
    name = g.get("id", "(무명)")
    where = g.get("where", "")
    print(f"\n{'='*78}\n[{name}] {where}\n{'='*78}")
    srcs = g.get("sources", [])
    metas, missing = {}, []
    for s in srcs:
        d = load_json(s)
        if d is None:
            missing.append(s); continue
        metas[s] = extract_meta(d)
    for m in missing:
        print(f"  **없음**: {m}")

    keys = sorted({k for m in metas.values() for k in m})
    if keys:
        w = max(len(Path(s).name) for s in metas) if metas else 10
        print(f"  {'산출물':<{w}}  " + "  ".join(f"{k}" for k in keys))
        for s, m in metas.items():
            print(f"  {Path(s).name:<{w}}  " +
                  "  ".join(str(m.get(k, "—")) for k in keys))

    allow = g.get("allow_differ", {}) or {}
    problems = []
    for k in keys:
        vals = {s: m[k] for s, m in metas.items() if k in m}
        if len(set(map(str, vals.values()))) <= 1:
            continue
        if k in allow:
            print(f"  · {k} 불일치 — 허용됨: {allow[k]}")
        else:
            problems.append((k, vals))

    for k, vals in problems:
        print(f"  **조건 불일치: {k}**")
        for s, v in vals.items():
            print(f"      {Path(s).name}: {v}")
        print(f"      → 같은 표/문단에서 읽히는 값들이 다른 {k}에서 왔다. "
              f"의도한 것이면 allow_differ에 이유를 적을 것.")

    # 값 대조
    vbad = 0
    for v in g.get("values", []) or []:
        ref, path = v["from"].split("::", 1)
        d = load_json(ref)
        if d is None:
            print(f"  **없음**: {ref}"); vbad += 1; continue
        try:
            x = dig(d, path)
        except Exception as e:
            print(f"  **경로 오류** {v['from']}: {e}"); vbad += 1; continue
        if v.get("expr"):
            x = eval(v["expr"], {"__builtins__": {"len": len}, "math": math}, {"x": x})
        exp, tol = v["expect"], v.get("tol", 0.005)
        if abs(x - exp) > tol:
            print(f"  **값 불일치** {v['claim']}: 데이터 {x:.4g} ≠ 문서 {exp}")
            vbad += 1
    if not problems and not vbad and not missing:
        print("  ✔ 조건 일치, 값 일치")
    return len(problems) + vbad + len(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docs/claim_provenance.yaml")
    ap.add_argument("--strict", action="store_true",
                    help="문제가 하나라도 있으면 종료코드 1")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.exists():
        print(f"매니페스트가 없다: {mpath}"); sys.exit(2)
    try:
        import yaml
        groups = yaml.safe_load(mpath.read_text())
    except ImportError:
        print("pyyaml이 없어 JSON으로 읽는다"); groups = json.loads(mpath.read_text())

    total = sum(check_group(g, args.strict) for g in groups)
    print(f"\n{'='*78}")
    print(f"문제 {total}건" if total else "문제 없음 — 모든 그룹이 같은 조건이고 값도 일치한다")
    print("=" * 78)
    if args.strict and total:
        sys.exit(1)


if __name__ == "__main__":
    main()
