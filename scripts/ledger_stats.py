#!/usr/bin/env python3
"""약점 원장 집계와 정합성 검사. 2026-10-02 신설.

**왜 스크립트인가.** 원장을 사람이 grep으로 세다 두 번 틀렸다. 대응표에 넣은 `~~L10~~`
표기가 「한계」 행으로 세어져 닫힌 한계가 8인데 14로 나왔다. 합본 집계(`build_thesis.py`)와
**같은 부류의 오류**다 — 패턴이 다른 맥락을 센다. 그래서 **구획 경계로 한정해** 센다.

**집계는 부산물이다.** 본래 목적은 이번에 수동으로 두 번 한 대조를 자동화하는 것이다 —
*"살아 있는 한계가 모두 대응표에 있는가"*와 *"논문 8.3 항목이 모두 대응표에 있는가"*.
그 대조가 없어서 한계 넷(L17~L20)이 원장에서 빠져 있었고, 8.4절이 대응표에서 빠져 있었다.

실행:
    python scripts/ledger_stats.py            # 집계 + 검사 (문제가 있으면 종료코드 1)
    python scripts/ledger_stats.py --quiet    # 검사만
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs/weakness_ledger.md"
CHAPTER = ROOT / "thesis_ch7_ch8_draft.md"

SECTIONS = {
    "closed": ("## 닫힘 — 자동 검사가", "## 한계 —"),
    "limits": ("## 한계 — 9장에", "### 논문 8.3절 ↔ 원장 대응"),
    "map": ("### 논문 8.3절 ↔ 원장 대응", "## 열림 — 진행"),
    "open": ("## 열림 — 진행", "## 분류 규칙"),
}


def slice_sections(text):
    out = {}
    for key, (start, end) in SECTIONS.items():
        i = text.index(start)
        out[key] = text[i:text.index(end, i)]
    return out


def rows(section):
    """표의 데이터 행만 — 머리행과 구분선(`|---|`)은 뺀다."""
    out = []
    for ln in section.split("\n"):
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        out.append(cells)
    return out


def ids(section, prefix):
    """구획에서 ID를 뽑는다. (살아 있는 것, 닫힌 것=취소선)"""
    live, closed = [], []
    for cells in rows(section):
        m = re.fullmatch(r"(?:\*\*)?(" + prefix + r"\d+(?:-old)?)(?:\*\*)?", cells[0])
        if m:
            live.append(m.group(1))
            continue
        m = re.fullmatch(r"~~(" + prefix + r"\d+(?:-old)?)~~", cells[0])
        if m:
            closed.append(m.group(1))
    return live, closed


def paper_limits():
    """논문 8.3절의 `**N. 제목**`과 8.4절의 `- **제목**`."""
    text = CHAPTER.read_text(encoding="utf-8")
    s83 = text[text.index("## 8.3 한계"):text.index("## 8.4 한계로 두지 않은 것")]
    s84 = text[text.index("## 8.4 한계로 두지 않은 것"):]
    nums = [int(m.group(1)) for m in re.finditer(r"^\*\*(\d+)\. ", s83, flags=re.M)]
    items = re.findall(r"^- \*\*([^*]+)\*\*", s84, flags=re.M)
    return nums, items


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--quiet", action="store_true", help="집계를 찍지 않고 검사만 한다")
    args = ap.parse_args()

    text = LEDGER.read_text(encoding="utf-8")
    sec = slice_sections(text)

    c_live, _ = ids(sec["closed"], "C")
    l_live, l_closed = ids(sec["limits"], "L")
    o_live, o_closed = ids(sec["open"], "O")

    # 대응표: 8.3 번호 / 거기 적힌 원장 L / 8.4 행 수
    map_rows = rows(sec["map"])
    map_83, map_L, map_84 = [], set(), 0
    for cells in map_rows:
        m = re.match(r"^(\d+)\. ", cells[0])
        if m:
            map_83.append(int(m.group(1)))
        elif cells[0] == "—" or re.match(r"^~~L", cells[0]):
            pass
        elif len(cells) >= 2 and ("행 없음" in cells[1] or re.fullmatch(r"L\d+|~~L\d+~~", cells[1])):
            map_84 += 1
        map_L |= {f"L{n}" for n in re.findall(r"L(\d+)", " ".join(cells[:2]))}

    if not args.quiet:
        print("약점 원장 집계 (구획 경계로 한정 — 대응표의 ~~L~~ 표기에 오염되지 않는다)\n")
        print(f"  닫힘 C            {len(c_live):3d}   {' '.join(c_live)}")
        print(f"  한계 살아있는 L   {len(l_live):3d}   {' '.join(l_live)}")
        print(f"  한계 닫힌 L       {len(l_closed):3d}   {' '.join(l_closed)}")
        print(f"  열림 살아있는 O   {len(o_live):3d}   {' '.join(o_live)}")
        print(f"  열림 닫힌 O       {len(o_closed):3d}")
        print(f"  대응표 데이터 행  {len(map_rows):3d}")
        print()

    problems = []

    # 1) 번호 중복 — 같은 ID를 두 번 쓰면 한쪽이 조용히 묻힌다
    for name, seq in (("C", c_live), ("L", l_live + l_closed), ("O", o_live + o_closed)):
        dup = {x for x in seq if seq.count(x) > 1}
        if dup:
            problems.append(f"{name} 번호 중복: {sorted(dup)}")

    # 2) 살아 있는 한계가 모두 대응표에 있는가 — 없어서 L17~L20이 빠져 있었다
    missing = [x for x in l_live if x not in map_L]
    if missing:
        problems.append(f"대응표에 없는 살아 있는 한계: {missing} — 논문의 어디에 적혀 있는지 밝힌다")

    # 3) 대응표가 가리키는 L이 원장에 실존하는가 (오타·삭제)
    ghost = sorted(map_L - set(l_live) - set(l_closed))
    if ghost:
        problems.append(f"대응표가 없는 한계를 가리킨다: {ghost}")

    # 4) 논문 8.3·8.4와 대응표가 맞는가
    nums, items84 = paper_limits()
    if sorted(nums) != sorted(map_83):
        problems.append(f"논문 8.3 항목 {sorted(nums)} ≠ 대응표 {sorted(map_83)}")
    if len(items84) != map_84:
        problems.append(f"논문 8.4 항목 {len(items84)}개 ≠ 대응표 {map_84}행 "
                        f"— 「한계가 아니라고 판정한 것」도 기록해야 다시 논쟁하지 않는다")

    # 5) 열림 행에 담당·비용이 있는가 (원장 규칙: 열림에는 비용과 담당을 적는다)
    for cells in rows(sec["open"]):
        if re.fullmatch(r"\*\*(O\d+)\*\*", cells[0]) and len(cells) >= 4:
            if cells[1].strip() in ("", "—") or cells[2].strip() in ("", "—") or cells[3].strip() in ("", "—"):
                problems.append(f"{cells[0]}에 담당·비용이 비어 있다 — 열림 항목의 규칙이다")

    print("─" * 72)
    if problems:
        print(f"✗ {len(problems)}건")
        for p in problems:
            print(f"    ✗ {p}")
        return 1
    print("문제 없음 — 번호 중복 없음, 살아 있는 한계 전부 대응표에 있음,")
    print("           대응표가 논문 8.3·8.4와 일치, 열림 항목에 담당·비용 있음.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
