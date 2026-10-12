#!/usr/bin/env python3
"""합본의 표를 한글에 붙여넣을 HTML로 뽑는다. 2026-10-12 신설.

**왜.** 조판에서 가장 오래 걸리는 일이 표 40개를 손으로 다시 그리는 것이다. 한글은
**HTML 표를 붙여넣으면 행·열 구조를 유지**하므로, 브라우저에서 표를 복사해 붙이면
칸을 하나씩 만들 필요가 없다.

**무엇을 지키는가.** 캡션은 **표 아래**(학교 양식)에 그대로 두고, 셀 안의 `**굵게**`와
`` `코드` ``를 각각 <b>, <code>로 바꾼다. 정렬 행(`|---|`)은 버린다.

실행:
    python scripts/export_tables.py              # → thesis_tables.html
    python scripts/export_tables.py --out X.html
"""

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cell(md):
    """셀 한 칸 — 이스케이프한 뒤 굵게·코드만 되살린다."""
    s = html.escape(md.strip())
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def tables(text):
    """(캡션, 행들) 목록. 합본은 표 뒤에 `**[표 N]** 제목`이 온다."""
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        rows = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            cells = [c for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells):
                rows.append([cell(c) for c in cells])
            i += 1
        cap = ""
        for j in range(i, min(i + 3, len(lines))):
            m = re.match(r"^\*\*\[표 (\d+)\]\*\* (.*)$", lines[j].strip())
            if m:
                cap = f"[표 {m.group(1)}] {cell(m.group(2))}"
                break
        if rows:
            out.append((cap, rows))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--src", default="thesis_full_draft.md")
    ap.add_argument("--out", default="thesis_tables.html")
    args = ap.parse_args()

    found = tables((ROOT / args.src).read_text(encoding="utf-8"))
    named = [(c, r) for c, r in found if c]
    parts = ["""<!doctype html>
<meta charset="utf-8">
<title>표 모음 — 한글 붙여넣기용</title>
<style>
 body { font-family: "맑은 고딕", sans-serif; margin: 28px; max-width: 1000px; }
 table { border-collapse: collapse; margin: 6px 0; }
 th, td { border: 1px solid #333; padding: 5px 9px; font-size: 13px; }
 tr:first-child td, th { background: #e8e8e8; font-weight: bold; }
 .cap { text-align: center; margin: 4px 0 26px; font-size: 13px; }
 .hint { background: #fffbe6; border: 1px solid #e8d98a; padding: 10px 14px;
         font-size: 13px; margin-bottom: 22px; }
</style>
<div class="hint"><b>쓰는 법.</b> 표를 드래그해 복사한 뒤 한글에 붙여넣으면 행·열이 그대로 들어갑니다.
캡션은 <b>표 아래 중앙</b>(학교 양식)에 두었습니다. 셀 안 서식은 굵게·고정폭만 남겼습니다.</div>"""]
    for cap, rows in named:
        parts.append("<table>")
        for r in rows:
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        parts.append("</table>")
        parts.append(f'<div class="cap">{cap}</div>')
    (ROOT / args.out).write_text("\n".join(parts) + "\n", encoding="utf-8")

    print(f"표 {len(named)}개 → {args.out}")
    if len(found) != len(named):
        print(f"  (캡션 없는 표 {len(found) - len(named)}개는 뺐다 — 본문 안 설명용 표다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
