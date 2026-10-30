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
    s = re.sub(r"`(.+?)`", r"\1", s)   # 2026-10-30: 한글에 붙일 때 고정폭 글꼴이 섞이지 않게 코드 표기는 글자로
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
        m = re.match(r"^\*\*<((?:부록 )?표) (\d+)>\*\* (.*)$", lines[j]) if j >= 0 else None
        if not m:
            continue                      # 캡션 없는 표(본문 중 보조 표·부록 표)는 건너뛴다
        num, title = f"{m.group(1)} {m.group(2)}", m.group(3)   # 「표 3」 또는 「부록 표 1」 (2026-10-30 부록 표 포함)
        # 조건 줄 `(250장, …)`은 표 바로 뒤에 온다
        k = i
        while k < len(lines) and not lines[k].strip():
            k += 1
        cond = lines[k].strip() if k < len(lines) and lines[k].strip().startswith("(") else ""
        out.append((num, title, cond, tbl))
    return out


# 2026-10-30: 한글(HWP)에 붙일 때 `<style>`의 서식은 따라가지 않는 경우가 있어 칸마다 인라인으로 박는다.
# 글꼴은 지정하지 않는다 — 한글 문서의 본문 글꼴을 그대로 따르게 한다
TD = "border:1px solid #000;padding:3pt 6pt;font-size:9pt;text-align:center;vertical-align:middle;"
TH = TD + "font-weight:bold;background:#e6e6e6;"
CAP = "text-align:center;font-size:10pt;margin:0 0 4pt 0;"
CON = "text-align:center;font-size:8.5pt;margin:4pt 0 0 0;"


def render(items):
    rows = []
    for k, (num, title, cond, tbl) in enumerate(items):
        body = []
        for n, ln in enumerate(tbl):
            if re.fullmatch(r"\|[-: |]+\|", ln.strip()):
                continue                  # 구분선
            tag, st = ("th", TH) if n == 0 else ("td", TD)
            tds = "".join(f'<{tag} style="{st}">{md_inline(c)}</{tag}>' for c in cells(ln))
            body.append(f"<tr>{tds}</tr>")
        cap = f'<p style="{CAP}">&lt;{num}&gt; {md_inline(title)}</p>'
        con = f'<p style="{CON}">{md_inline(cond)}</p>' if cond else ""
        tab = '<table style="border-collapse:collapse;margin:0 auto;">\n' + "\n".join(body) + "\n</table>"
        rows.append(f'<div class="bar"><button onclick="cp(\'b{k}\', this)">복사</button> &lt;{num}&gt;</div>\n'
                    f'<div class="blk" id="b{k}">{cap}\n{tab}\n{con}</div>')   # 제목은 표 위
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
    n_cap = len(re.findall(r"^\*\*<(?:부록 )?표 \d+>\*\*", text, flags=re.M))
    if len(items) != n_cap:
        raise SystemExit(f"표 수 불일치: 뽑은 {len(items)} ≠ 합본의 캡션 {n_cap} — "
                         "캡션이 표 바로 앞에 오지 않는 자리가 있는지 본다")
    for kind in ("표", "부록 표"):
        nums = [int(n.split()[-1]) for n, *_ in items if n.rsplit(" ", 1)[0] == kind]
        if nums != list(range(1, len(nums) + 1)):
            raise SystemExit(f"{kind} 번호가 1..N 연속이 아니다: {nums}")
    n_body = sum(1 for n, *_ in items if n.startswith("표 "))

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>표 모음 — 한글 붙여넣기용</title>
<style>
 body {{ font-family: "맑은 고딕", sans-serif; margin: 28px; max-width: 1000px; }}
 .bar {{ margin: 30px 0 6px; font-size: 13px; color: #555; }}
 .bar button {{ font-size: 13px; padding: 3px 12px; cursor: pointer; }}
 .blk {{ padding: 4px 0; }}
 .hint {{ background: #fffbe6; border: 1px solid #e8d98a; padding: 10px 14px;
          font-size: 13px; margin-bottom: 22px; line-height: 1.6; }}
</style>
<script>
function cp(id, btn) {{
  const el = document.getElementById(id), r = document.createRange(), sel = window.getSelection();
  r.selectNodeContents(el); sel.removeAllRanges(); sel.addRange(r);
  let ok = false; try {{ ok = document.execCommand("copy"); }} catch (e) {{}}
  sel.removeAllRanges();
  const t = btn.textContent; btn.textContent = ok ? "복사됨 ✓" : "Ctrl+C로 복사"; setTimeout(() => btn.textContent = t, 1500);
  if (!ok) {{ sel.addRange(r); }}
}}
</script>
<div class="hint">
 <b>자동 생성</b> — scripts/build_tables_html.py가 {args.src}에서 뽑았다. 직접 고치지 말고 합본을 고친 뒤 다시 생성한다.<br>
 <b>사용법</b>: 표 위의 <b>복사</b> 버튼을 누르면 <b>캡션·표·조건 줄</b>이 한 번에 복사된다. 한글(HWP)에서
 <b>Ctrl+V</b>로 붙이면 표 서식(테두리·머리 칸 음영·가운데 정렬)이 함께 넘어온다. 글꼴은 지정하지 않았으므로
 한글 문서의 본문 글꼴을 따른다. 붙인 뒤 표 전체를 선택해 글꼴만 맞추면 된다.<br>
 캡션은 표 위 중앙(학교 작성 방법), 표 아래 작은 글씨가 측정 조건이다 — <b>조건을 빼고 옮기면 값이 무엇을 잰 것인지 알 수 없다.</b>
</div>
<p>본문 표 {n_body}개 · 부록 표 {len(items) - n_body}개</p>

{render(items)}
"""
    (ROOT / args.out).write_text(doc, encoding="utf-8")
    n_cond = sum(1 for *_, c, _ in [(a, b, c, d) for a, b, c, d in items] if c)
    print(f"본문 표 {n_body}개 · 부록 표 {len(items) - n_body}개 · 조건 줄 {n_cond}개 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
