#!/usr/bin/env python3
"""수치 전수 감사 — 폐기된 값이 본문에 남아 있는지 찾는다.

**왜 필요한가.** 이 저장소에서 반복해 나온 오류 유형은 "값을 고쳤는데 그 값이 사는
다른 자리를 안 고친" 것이다(`thesis_outline_20260910.md` §3.5). 이 유형은 **값 대조로
잡히지 않는다** — 옛 값도 원래 자리에서는 맞는 값이고, 틀린 것은 출처뿐이기 때문이다.
실제 사례:

  - §6.3이 "1.34~1.38배로 적는다"고 지시해 놓고 **바로 아래 세 문단이 1.41배**를 썼다
  - §7.3이 비용을 +0.78%p로 대체했는데 **같은 장 첫머리가 −0.59%p**를 들고 있었다
  - 본문을 3.2%로 고쳤는데 **Fig. 1 주석이 "모든 변형이 0%"** 그대로였다

그래서 정정 박스와 본문을 **구분해서** 본다. 정정 박스가 옛 값을 인용하는 것은 정상이고,
본문이 들고 있는 것만 오류다.

사용법
------
    python scripts/audit_numbers.py                 # 전체 감사 (오류 있으면 exit 1)
    python scripts/audit_numbers.py --all           # 폐기·주의·현행 전부 나열
    python scripts/audit_numbers.py --show "1.41배" # 특정 문자열의 모든 출현 위치
    python scripts/audit_numbers.py --files a.md b.md

대장은 `number_registry.json`이다. **값을 정정할 때 대장에 먼저 적고 감사를 돌린다.**
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "number_registry.json"

# 기본 감사 대상 — 수치가 실리는 문서들
DEFAULT_GLOBS = ("*.md", "*.html", "*.txt")
SKIP_NAMES = {"CLAUDE.md", "number_registry.json"}

# 이 표식이 있으면 그 줄은 "기록"이다 — 옛 값을 인용해도 정상
RECORD_MARKERS = re.compile(
    r"정정|철회|이전 판|초판|폐기|대체(했|한|된|됐)|틀렸다|낡은|자기정정|"
    r"사전 등록|결과 보기 전|라고만 썼|적혀 있었|적고 있었|들고 있었|"
    r"쓰지 않는다|쓰지 말|말하지 않는다|금지|조건 혼입|~~|->|→"
)
# 이 제목 아래는 통째로 기록 구역 (번호 접두사 허용)
RECORD_SECTIONS = re.compile(
    r"^#+\s*[\d.]*\s*(작성 메모|.*사전 등록|.*분기$|제출본 수정 필요 항목|"
    r"제출본 대비|디스크 정리 기록|남은 할 일|수치 정정 규칙)"
)
HEADING = re.compile(r"^#{1,6}\s")


def classify(lines):
    """줄마다 '본문' / '기록'을 매긴다.

    기록 구역은 **하위 제목으로 끝나지 않는다** — 「제출본 수정 필요 항목」 아래의
    `### ① …` 같은 하위 절도 같은 구역이다. 구역을 연 제목과 **같거나 더 높은 수준**의
    제목이 나올 때만 구역이 닫힌다.
    """
    kinds, in_fence = [], False
    record_depth = None                 # 기록 구역을 연 제목의 깊이 (None이면 구역 밖)
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            kinds.append("기록" if in_fence else "본문")
            continue
        m = HEADING.match(s)
        if m:
            depth = len(s) - len(s.lstrip("#"))
            if RECORD_SECTIONS.match(s):
                record_depth = depth
            elif record_depth is not None and depth <= record_depth:
                record_depth = None
        in_record_section = record_depth is not None
        if in_fence or in_record_section:
            kinds.append("기록")
        elif s.startswith(">"):            # 인용 = 이 저장소의 정정 박스 관례
            kinds.append("기록")
        elif RECORD_MARKERS.search(ln):
            kinds.append("기록")
        else:
            kinds.append("본문")
    return kinds


NUMERIC = re.compile(r"^[\d,.]+")


def make_matcher(needle):
    """숫자로 시작하는 needle은 **앞뒤에 숫자·점이 붙지 않을 때만** 일치시킨다.
    이것이 없으면 `50.73%`가 `0.73`으로, `2602.07596`이 `2.07`로 잡힌다."""
    if NUMERIC.match(needle):
        pat = re.compile(r"(?<![\d.])" + re.escape(needle) + r"(?![\d])")
        return lambda ln: bool(pat.search(ln))
    return lambda ln: needle in ln


def scan(paths, needles):
    """needle -> [(path, lineno, kind, line)]"""
    hits = {n: [] for n in needles}
    match = {n: make_matcher(n) for n in needles}
    for p in paths:
        try:
            lines = p.read_text(encoding="utf-8").split("\n")
        except (UnicodeDecodeError, OSError):
            continue
        kinds = classify(lines)
        for i, (ln, kind) in enumerate(zip(lines, kinds), 1):
            for n in needles:
                if match[n](ln):
                    hits[n].append((p, i, kind, ln.strip()))
    return hits


def collect_files(args):
    if args.files:
        return [Path(f) for f in args.files]
    out = []
    for g in DEFAULT_GLOBS:
        out += [p for p in ROOT.glob(g) if p.name not in SKIP_NAMES]
    return sorted(out)


def rel(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="현행·주의 값의 출현 위치도 전부 나열")
    ap.add_argument("--show", help="이 문자열의 모든 출현 위치를 보여준다 (대장과 무관)")
    ap.add_argument("--files", nargs="*", help="감사 대상 파일 (미지정이면 저장소 루트의 md/html/txt)")
    args = ap.parse_args()

    files = collect_files(args)
    if not files:
        print("감사할 파일이 없다."); return 1

    if args.show:
        for p, i, kind, ln in scan(files, [args.show])[args.show]:
            print(f"  [{kind}] {rel(p)}:{i}  {ln[:100]}")
        return 0

    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    print(f"감사 대상 {len(files)}개 파일 · 대장 {len(reg['entries'])}개 항목\n")

    errors, warns = [], []
    for e in reg["entries"]:
        retired = e.get("retired", [])
        watch = e.get("watch", [])
        canon = e.get("canonical", [])
        hits = scan(files, retired + watch + (canon if args.all else []))

        bad = [(n, h) for n in retired for h in hits[n] if h[2] == "본문"]
        rec = [(n, h) for n in retired for h in hits[n] if h[2] == "기록"]
        wat = [(n, h) for n in watch for h in hits[n] if h[2] == "본문"]

        if bad:
            errors.append((e, bad))
        if wat:
            warns.append((e, wat))

        mark = "✗" if bad else ("△" if wat else "✓")
        print(f"{mark} {e['name']}")
        print(f"    현행 {', '.join(canon) if canon else '—'}"
              + (f"   조건: {e['condition']}" if e.get("condition") else ""))
        if retired:
            print(f"    폐기 {', '.join(retired)}  →  본문 {len(bad)}건 / 기록 {len(rec)}건")
        for n, (p, i, _k, ln) in bad:
            print(f"      ✗ {rel(p)}:{i}  «{n}»  {ln[:78]}")
        for n, (p, i, _k, ln) in wat:
            print(f"      △ {rel(p)}:{i}  «{n}»  {ln[:78]}")
        if args.all:
            for n in canon:
                for p, i, kind, ln in hits.get(n, []):
                    if kind == "본문":
                        print(f"      · {rel(p)}:{i}  «{n}»  {ln[:74]}")
        print()

    print("─" * 72)
    if errors:
        print(f"✗ 폐기된 값이 본문에 {sum(len(b) for _, b in errors)}건 남아 있다.")
        print("  정정 박스 안이면 정상이다 — 그 경우 RECORD_MARKERS에 표식을 추가할 것.")
    if warns:
        print(f"△ 조건 병기가 필요한 값이 본문에 {sum(len(w) for _, w in warns)}건 있다.")
        print("  틀린 값이 아니다. 조건(표본·통계량·해상도)이 함께 적혀 있는지만 확인할 것.")
    if not errors and not warns:
        print("✓ 폐기된 값 없음.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
