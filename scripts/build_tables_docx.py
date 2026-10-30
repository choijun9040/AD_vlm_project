#!/usr/bin/env python3
"""합본의 표를 Word(.docx)로 뽑는다 — 한글(HWP)로 옮기기용. 2026-10-30 신설.

브라우저에서 HTML 표를 복사해 한글에 붙이면 환경에 따라 칸이 하나로 합쳐진다. 한글은 .docx를 열 때
표를 한글 표로 변환하므로, 이 파일을 한글에서 열고 표를 복사해 논문 문서에 붙이면 칸이 깨지지 않는다.
python-docx 없이 표준 라이브러리로 최소 구조(document.xml)만 쓴다 — 공용 환경에 패키지를 더하지 않는다.

표·캡션·조건 줄은 `build_tables_html.extract()`로 합본에서 같은 규칙으로 뽑는다(짝이 어긋나지 않게).

실행:
    python scripts/build_tables_docx.py            # → thesis_tables.docx
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tables_html import ROOT, cells, extract   # noqa: E402

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def runs(text, size, bold=False):
    """`**굵게**`만 살리고 코드 표기(`…`)는 글자로. size는 반 포인트 단위."""
    out = []
    for k, part in enumerate(re.split(r"\*\*(.+?)\*\*", text.replace("`", ""))):
        if not part:
            continue
        b = "<w:b/>" if (bold or k % 2 == 1) else ""
        out.append(f'<w:r><w:rPr>{b}<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>'
                   f'<w:t xml:space="preserve">{escape(part)}</w:t></w:r>')
    return "".join(out)


def para(text, size, align="center", bold=False, before=0, after=0):
    return (f'<w:p><w:pPr><w:spacing w:before="{before}" w:after="{after}"/><w:jc w:val="{align}"/></w:pPr>'
            f'{runs(text, size, bold)}</w:p>')


def table(rows):
    border = "".join(f'<w:{s} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                     for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    body = []
    for n, ln in enumerate(rows):
        if re.fullmatch(r"\|[-: |]+\|", ln.strip()):
            continue                                   # 구분선
        head = n == 0
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="E6E6E6"/>' if head else ""
        tcs = "".join(f'<w:tc><w:tcPr><w:vAlign w:val="center"/>{shd}</w:tcPr>{para(c, 18, bold=head)}</w:tc>'
                      for c in cells(ln))
        hdr = '<w:trPr><w:tblHeader/></w:trPr>' if head else ""
        body.append(f"<w:tr>{hdr}{tcs}</w:tr>")
    return (f'<w:tbl><w:tblPr><w:jc w:val="center"/><w:tblBorders>{border}</w:tblBorders>'
            f'<w:tblLayout w:type="autofit"/></w:tblPr>' + "".join(body) + "</w:tbl>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="thesis_full_draft.md")
    ap.add_argument("--out", default="thesis_tables.docx")
    args = ap.parse_args()
    items = extract((ROOT / args.src).read_text(encoding="utf-8"))
    parts = []
    for num, title, cond, rows in items:
        parts.append(para(f"<{num}> {title}", 20, before=360, after=80))   # 제목은 표 위 (학교 작성 방법)
        parts.append(table(rows))
        if cond:
            parts.append(para(cond, 17, before=60))
    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {W}><w:body>'
           + "".join(parts)
           + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
             '<w:pgMar w:top="1418" w:right="1134" w:bottom="1418" w:left="1134"/></w:sectPr></w:body></w:document>')
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Override PartName="/word/document.xml" '
          'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(ROOT / args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", doc)
    n_body = sum(1 for n, *_ in items if n.startswith("표 "))
    print(f"본문 표 {n_body}개 · 부록 표 {len(items) - n_body}개 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
