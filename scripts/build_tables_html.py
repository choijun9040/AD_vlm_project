#!/usr/bin/env python3
"""합본의 표를 한글 붙여넣기용 HTML로 뽑는다. 2026-10-12 신설.

**왜 합본에서 뽑는가.** 표와 캡션을 따로 모으면 **짝이 어긋난다.** 실제로 그 일이 일어났다 —
당시 양식(졸업생 논문)은 캡션이 **표 아래**였는데 `캡션 → 표`로 읽은 파일이 만들어져 **40개 제목이 한 칸씩
밀려** 있었다(표 14 캡션에 표 15 내용). 여기서는 합본을 순서대로 훑으며 **표를 먼저 모으고
바로 뒤의 캡션을 붙이므로** 구조적으로 어긋날 수 없다.

**조건 줄도 함께 가져간다.** 합본은 표 다음에 `(250장, TensorRT 10.13, A100 …)`처럼 측정
조건을 적는다. 이것이 빠지면 **값은 맞는데 무엇을 잰 것인지 알 수 없다**(원장 C8).

실행:
    python scripts/build_tables_html.py            # → thesis_tables_rebuilt.html
    python scripts/build_tables_html.py --out X.html
"""

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def md_inline(s):
    """표 칸의 마크다운 강조만 HTML로. 나머지는 글자 그대로 둔다."""
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def extract(text):
    """(번호, 제목, 조건, 표 행들) 목록. 합본 순서를 그대로 따른다."""
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        start = i
        tbl = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            tbl.append(lines[i])
            i += 1
        # 2026-10-27 학교 작성 방법: 캡션은 표 **위**다 — 표 앞의 빈 줄을 건너뛰고 캡션을 찾는다
        j = start - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        m = re.match(r"^\*\*<표 (\d+)>\*\* (.*)$", lines[j]) if j >= 0 else None
        if not m:
            continue                      # 캡션 없는 표(본문 중 보조 표·부록 표)는 건너뛴다
        num, title = m.group(1), m.group(2)
        # 조건 줄 `(250장, …)`은 표 바로 뒤에 온다
        k = i
        while k < len(lines) and not lines[k].strip():
            k += 1
        cond = lines[k].strip() if k < len(lines) and lines[k].strip().startswith("(") else ""
        out.append((num, title, cond, tbl))
    return out


def render(items):
    rows = []
    for num, title, cond, tbl in items:
        body = []
        for n, ln in enumerate(tbl):
            if re.fullmatch(r"\|[-: |]+\|", ln.strip()):
                continue                  # 구분선
            tag = "th" if n == 0 else "td"
            tds = "".join(f"<{tag}>{md_inline(c)}</{tag}>" for c in cells(ln))
            body.append(f"<tr>{tds}</tr>")
        cap = f'<div class="cap">&lt;표 {num}&gt; {md_inline(title)}</div>'
        con = f'<div class="cond">{md_inline(cond)}</div>' if cond else ""
        rows.append(cap + f"\n<table>\n" + "\n".join(body) + "\n</table>\n" + con)   # 제목은 표 위
    return "\n\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--src", default="thesis_full_draft.md")
    ap.add_argument("--out", default="thesis_tables_rebuilt.html")
    args = ap.parse_args()

    text = (ROOT / args.src).read_text(encoding="utf-8")
    items = extract(text)
    n_cap = len(re.findall(r"^\*\*<표 \d+>\*\*", text, flags=re.M))
    if len(items) != n_cap:
        raise SystemExit(f"표 수 불일치: 뽑은 {len(items)} ≠ 합본의 캡션 {n_cap} — "
                         "캡션이 표 바로 앞에 오지 않는 자리가 있는지 본다")
    nums = [int(n) for n, *_ in items]
    if nums != sorted(nums) or nums != list(range(1, len(nums) + 1)):
        raise SystemExit(f"표 번호가 1..N 연속이 아니다: {nums}")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>표 모음 — 한글 붙여넣기용</title>
<style>
 body {{ font-family: "맑은 고딕", sans-serif; margin: 28px; max-width: 1000px; }}
 table {{ border-collapse: collapse; margin: 6px 0; }}
 th, td {{ border: 1px solid #333; padding: 5px 9px; font-size: 13px; }}
 th {{ background: #e8e8e8; font-weight: bold; }}
 .cap {{ text-align: center; margin: 6px 0 2px; font-size: 13px; }}
 .cond {{ text-align: center; margin: 0 0 26px; font-size: 12px; color: #444; }}
 .hint {{ background: #fffbe6; border: 1px solid #e8d98a; padding: 10px 14px;
          font-size: 13px; margin-bottom: 22px; line-height: 1.6; }}
</style>
<div class="hint">
 <b>자동 생성</b> — <code>scripts/build_tables_html.py</code>가 <code>{args.src}</code>에서 뽑았다.
 직접 고치지 말고 합본을 고친 뒤 다시 생성한다.<br>
 캡션은 <b>표 위 중앙</b>(학교 작성 방법)이고, 표 아래 작은 글씨가 <b>측정 조건</b>이다 —
 <b>조건을 빼고 옮기면 값이 무엇을 잰 것인지 알 수 없게 된다.</b><br>
 표를 끌어 선택해 한글에 붙이면 표 서식이 함께 넘어간다. 캡션과 조건은 따로 복사한다.
</div>
<p>표 {len(items)}개</p>

{render(items)}
"""
    (ROOT / args.out).write_text(doc, encoding="utf-8")
    n_cond = sum(1 for *_, c, _ in [(a, b, c, d) for a, b, c, d in items] if c)
    print(f"표 {len(items)}개 (번호 1..{len(items)} 연속) · 조건 줄 {n_cond}개 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
