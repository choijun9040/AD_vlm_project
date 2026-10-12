#!/usr/bin/env python3
"""합본의 표를 한글 붙여넣기용 HTML로 뽑는다. 2026-10-12 신설.

**왜 스크립트인가.** 손으로 옮긴 판(`thesis_tables.html`)에서 **캡션이 한 칸씩 밀려** 모든 표의
제목이 다음 표에 붙어 있었다. 원인은 합본이 **`표 → 캡션`**(학교 양식: 캡션은 표 아래)인데
**`캡션 → 표`**로 읽은 것이다. 여기서는 **캡션 줄을 기준으로 삼아 그 앞의 표를 집으므로**
같은 방식으로 어긋날 수 없고, 캡션 수와 표 수가 다르면 멈춘다.

**조건 줄도 함께 옮긴다.** 합본은 캡션 다음 줄에 `(250장, 평가 해상도, bf16)` 같은 측정 조건을
둔다. 그것이 빠지면 값은 맞는데 **무엇을 잰 것인지 알 수 없다** — 원장 C8이 막으려는 부류다.

실행:
    python scripts/build_tables_html.py            # → thesis_tables_rebuilt.html
    python scripts/build_tables_html.py --out X    # 다른 이름으로
"""

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "thesis_full_draft.md"

STYLE = """ body { font-family: "맑은 고딕", sans-serif; margin: 28px; max-width: 1000px; }
 table { border-collapse: collapse; margin: 6px 0; }
 th, td { border: 1px solid #333; padding: 5px 9px; font-size: 13px; vertical-align: top; }
 tr:first-child td { background: #e8e8e8; font-weight: bold; }
 .cap { text-align: center; margin: 4px 0 2px; font-size: 13px; }
 .cond { text-align: center; margin: 0 0 26px; font-size: 12px; color: #444; }
 .hint { background: #fffbe6; border: 1px solid #e8d98a; padding: 10px 14px;
         font-size: 13px; margin-bottom: 22px; line-height: 1.6; }"""

HINT = """<div class="hint">
<b>한글 붙여넣기용 표 모음</b> — <code>scripts/build_tables_html.py</code>가
<code>thesis_full_draft.md</code>에서 뽑았다. 직접 고치지 말고 합본을 고친 뒤 다시 돌린다.<br>
<b>캡션은 표 아래</b>이고(학교 양식) 그 아래 작은 글씨가 <b>측정 조건</b>이다 —
조건을 빼고 옮기면 값이 무엇을 잰 것인지 알 수 없게 된다.<br>
표는 브라우저에서 드래그해 복사하면 한글에 표로 들어간다.
</div>"""


def cells(row):
    out = [c.strip() for c in row.strip().strip("|").split("|")]
    return [re.sub(r"\*\*(.+?)\*\*", r"\1", re.sub(r"`(.+?)`", r"\1", c)) for c in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--out", default="thesis_tables_rebuilt.html")
    args = ap.parse_args()

    lines = SRC.read_text(encoding="utf-8").split("\n")
    blocks = []
    for i, ln in enumerate(lines):
        m = re.match(r"^\*\*\[표 (\d+)\]\*\* (.*)$", ln)
        if not m:
            continue
        num, title = m.group(1), m.group(2)

        # 캡션 **앞**의 표 블록을 집는다 — 이 방향이라야 한 칸 밀릴 수 없다
        j = i - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        tbl = []
        while j >= 0 and lines[j].lstrip().startswith("|"):
            tbl.append(lines[j])
            j -= 1
        tbl.reverse()
        if not tbl:
            raise SystemExit(f"[표 {num}]의 캡션 앞에 표가 없다 (합본 {i + 1}행)")

        # 캡션 **뒤**의 조건 줄 — `(`로 시작하는 한 줄
        k = i + 1
        while k < len(lines) and not lines[k].strip():
            k += 1
        cond = lines[k].strip() if k < len(lines) and lines[k].lstrip().startswith("(") else ""
        blocks.append((num, title, tbl, cond))

    n_cap = len(blocks)
    n_tbl = sum(1 for ln in lines if re.match(r"^\s*\|", ln) and "---" in ln)
    if n_cap != n_tbl:
        raise SystemExit(f"캡션 {n_cap}개 ≠ 표 {n_tbl}개 — 캡션이 없는 표가 있는지 본다")

    out = ['<!doctype html>', '<meta charset="utf-8">',
           '<title>표 모음 — 한글 붙여넣기용</title>', f"<style>\n{STYLE}\n</style>", HINT]
    for num, title, tbl, cond in blocks:
        rows = [r for r in tbl if "---" not in r or not set(r.replace("|", "").strip()) <= set("- :")]
        out.append("<table>")
        for r in rows:
            cs = "".join(f"<td>{html.escape(c)}</td>" for c in cells(r))
            out.append(f"<tr>{cs}</tr>")
        out.append("</table>")
        out.append(f'<div class="cap">[표 {num}] {html.escape(title)}</div>')
        if cond:
            out.append(f'<div class="cond">{html.escape(cond)}</div>')
        else:
            out.append('<div class="cond"></div>')

    (ROOT / args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    n_cond = sum(1 for b in blocks if b[3])
    print(f"표 {n_cap}개 · 조건 줄 {n_cond}개 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
