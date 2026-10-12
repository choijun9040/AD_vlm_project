"""
학위 논문 합본 생성 — 장별 초안을 하나의 원고(thesis_full_draft.md)로 묶는다.

**생성물은 직접 고치지 않는다.** 고칠 것은 장별 초안(thesis_ch*_draft.md, thesis_abstract_draft.md,
thesis_references_draft.md, thesis_figures.md)이고, 이 스크립트를 다시 돌리면 합본이 따라온다.

하는 일
  1. 각 초안의 머리말(첫 `---`까지의 작성 메모)을 떼고 본문만 잇는다.
  2. **내부 정정 기록을 뺀다** — 첫 줄에 "정정"이 들어간 인용 블록(`> **정정 …`, `> *정정 기록.*`).
     작업 기록으로는 필요하지만 제출 원고의 본문은 아니다. (개수를 출력한다)
  3. **인용 번호를 매긴다** — 본문에 처음 나오는 순서. arXiv 표기 "(arXiv:XXXX.XXXXX, …)"는 [n]으로 바꾸고,
     모델·데이터셋·도구는 **장마다 처음 나올 때** 한 번 인용한다. 제목·코드 블록·코드 스팬 안은 건드리지 않는다.
     참고문헌 목록은 새 번호 순서로 다시 적는다. 인용되지 않은 항목은 경고한다.
  4. **그림을 넣는다** — 기준 줄(표 아래 조건 줄) 뒤에 그림과 캡션(thesis_figures.md)을 넣고, 기준 줄에
     "그림 X-Y"를 붙여 본문이 그림을 가리키게 한다.
  5. 목차·그림 목차·표 목차를 만든다(합본 앞과 `thesis_toc.md` 따로)(그림 제목은 캡션 첫 문장, 표는 `**표 [키].** 제목`에 장별 번호, 본문 `[표:키]`는 번호로).

실행:
    python scripts/build_thesis.py            # → thesis_full_draft.md
    python scripts/build_thesis.py --check    # 쓰지 않고 통계만
"""

import argparse
import datetime as dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ["thesis_ch1_ch3_draft.md", "thesis_ch4_ch5_draft.md", "thesis_ch6_draft.md",
            "thesis_ch7_ch8_draft.md", "thesis_ch9_draft.md"]

# ── 인용: 참고문헌 초안의 (옛) 번호 → 본문에서 그 문헌을 가리키는 표기 ────────────────────
ARXIV = {"2601.21288": 3, "2303.06884": 6, "2305.04526": 7, "2605.01330": 8, "2407.08044": 9,
         "2502.09003": 10, "2403.06497": 11, "2402.17762": 12, "2605.15572": 13, "2510.04547": 14,
         "2607.08029": 15, "2503.01873": 16, "2604.26857": 18, "2506.24044": 19, "2512.16760": 20,
         "2510.24795": 21, "2608.30144": 22}
# 장마다 처음 한 번 인용한다. (정규식, 옛 번호). 뒤에 붙는 모델 크기 등(-7B, -Instruct)은 토큰째 넘긴다.
NAMED = [
    (r"Qwen2\.5-VL", 1), (r"FitNets", 2), (r"EM-VLM4AD", 4), (r"MiniDrive", 5),
    (r"vLLM (?:이슈 )?#40290", 17), (r"(?<![A-Za-z])nuScenes(?![A-Za-z-])", 23), (r"DriveLM", 24),
    (r"NuScenes-QA", 25), (r"CODA-LM", 26), (r"(?<![A-Za-z])CODA(?![A-Za-z-])", 27),
    (r"(?<![A-Za-z])LoRA(?![A-Za-z])", 28), (r"지식 증류", 29), (r"Spearman", 30),
    (r"Qwen2-VL", 31), (r"McNemar", 32), (r"(?<![A-Za-z-])AWQ(?![A-Za-z-])", 33), (r"Qwen3-VL", 34),
    (r"Phi-3\.5", 35), (r"LLaVA", 36), (r"InternVL3", 37), (r"SmolVLM", 38), (r"Pixtral", 39),
    (r"GLM-4\.1V", 40), (r"Kimi-VL", 41), (r"(?<![A-Za-z])DLA(?![A-Za-z])", 42),
    (r"Orin Nano", 43), (r"TensorRT 11\.0", 44),
]
TOKEN_TAIL = r"[A-Za-z0-9.\-]*"

# ── 그림: (일련번호, 캡션 키, 파일, 기준 줄이 포함하는 글자) ────────────────────────────
# **번호는 전체 일련이다 (2026-10-12, 학교 양식).** 졸업생 논문(2026-08)이 `[그림 1]`~`[그림 19]`로
# 장과 무관하게 이어 붙인다. 캡션 키("4-1" 등)는 `thesis_figures.md` 쪽 식별자라 그대로 둔다 —
# 번호와 키를 분리해야 그림을 끼워 넣어도 캡션 파일을 건드리지 않는다.
FIGURES = [
    ("1", "4-1", "figures/fig1_activation_profile.png", "(799장, 평가 해상도, bf16, 이미지별 최대의 중앙값)"),
    ("2", "4-2", "figures/fig2_resolution_collapse.png", "(fp16 붕괴율, 250장, 이미지 단위로 출력이 NaN이면 붕괴)"),
    ("3", "4-3", "figures/fig4_metric_inversion.png", "(원본 해상도, 250장, 마지막 블록, 여유는 p50·bf16 측정)"),
    ("4", "4-4", "figures/fig3_lambda_dose_response.png", "(799장, 평가 해상도, bf16, p50)"),
    ("5", "6-1", "figures/fig6_runtime_collapse.png", "(250장, 같은 순서, 15W 모드, TensorRT 10.3, 강타입. 지연은 두 분할 엔진의 합)"),
    ("6", "7-1", "figures/fig_gate_calibration.png", "최악 여유 1.00에서 붕괴 2%, 1.05에서 0%다. 따로 보정할 상수가 없다."),
]


def body_of(path):
    """첫 `---` 뒤부터가 본문이다(그 앞은 작성 메모)."""
    text = (ROOT / path).read_text()
    i = text.index("\n---\n")
    return text[i + 5:].strip("\n")


def drop_correction_quotes(text):
    """첫 줄에 '정정'이 있는 인용 블록을 통째로 뺀다."""
    out, lines, i, dropped = [], text.split("\n"), 0, 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(">") and "정정" in ln:
            dropped += 1
            while i < len(lines) and lines[i].startswith(">"):
                i += 1
            if out and out[-1] == "" and i < len(lines) and lines[i] == "":
                i += 1                      # 빈 줄이 겹치지 않게
            continue
        out.append(ln)
        i += 1
    return "\n".join(out), dropped


def protect(line):
    """코드 스팬·링크 주소를 가려 치환에서 빼고, 되돌리는 함수를 준다."""
    saved = []
    def keep(m):
        saved.append(m.group(0))
        return f"\x00{len(saved) - 1}\x00"
    line = re.sub(r"`[^`]*`|\]\([^)]*\)|https?://\S+", keep, line)
    return line, lambda s: re.sub(r"\x00(\d+)\x00", lambda m: saved[int(m.group(1))], s)


def cite_chapter(text):
    """한 장 안에서 인용 표지(옛 번호)를 넣는다: «R12» 꼴의 임시 표지."""
    text = text.replace("[44]", "")                      # 초안에 손으로 넣은 표지는 규칙으로 다시 단다
    seen, out, in_fence = set(), [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
        if in_fence or line.startswith("#"):
            out.append(line)
            continue
        s, restore = protect(line)
        for aid, n in ARXIV.items():                     # arXiv 표기는 매번 번호로 바꾼다
            s = re.sub(r"\(arXiv:%s(?:, [^)]*)?\)" % re.escape(aid), f"«R{n}»", s)
            s = re.sub(r", arXiv:%s" % re.escape(aid), f"«R{n}»", s)
            s = re.sub(r"arXiv:%s" % re.escape(aid), f"«R{n}»", s)
            if f"«R{n}»" in s:
                seen.add(n)
        for pat, n in NAMED:                             # 이름은 장마다 처음 한 번
            if n in seen:
                continue
            m = re.search(pat + TOKEN_TAIL, s)
            if m:
                end = m.end()
                while end > m.start() and s[end - 1] in ".-":
                    end -= 1
                s = s[:end] + f"«R{n}»" + s[end:]
                seen.add(n)
        out.append(restore(s))
    # 줄머리의 표지는 앞 줄 끝에 붙인다 — 원문이 "(arXiv:…)"를 다음 줄에 둔 경우 "이름 [n]"처럼 띄어진다
    return re.sub(r"\n«R", "«R", "\n".join(out))


def parse_references():
    """참고문헌 초안의 번호 목록을 {옛 번호: 본문} 으로."""
    text = (ROOT / "thesis_references_draft.md").read_text()
    body = text.split("\n---\n")[1]
    refs, cur = {}, None
    for ln in body.split("\n"):
        m = re.match(r"^(\d+)\. (.*)", ln)
        if m:
            cur = int(m.group(1)); refs[cur] = m.group(2).strip()
        elif cur and ln.startswith("    "):
            refs[cur] += " " + ln.strip()
        elif ln.startswith("## "):
            break
    return refs


def parse_captions():
    text = (ROOT / "thesis_figures.md").read_text()
    caps = {}
    for m in re.finditer(r"^\*\*그림 (\d-\d)\.\*\* (.*?)(?=\n\n)", text, flags=re.M | re.S):
        caps[m.group(1)] = " ".join(m.group(2).split())
    return caps


def insert_figures(text, caps):
    """그림을 기준 줄 뒤에 넣고 **캡션은 그림 아래**에 둔다 (학교 양식).

    캡션 표기는 `**[그림 N]** …`이다 — 졸업생 논문(2026-08)이 `[그림 1]` 형태를 쓴다.
    """
    for num, key, path, anchor in FIGURES:
        i = text.find(anchor)
        if i < 0:
            raise SystemExit(f"그림 {num}의 기준 줄을 찾지 못했다: {anchor[:40]}")
        j = text.index("\n", i)
        line = text[text.rfind("\n", 0, i) + 1:j]
        tag = f" ([그림 {num}])" if not line.rstrip().endswith(")") else ""
        new_line = (line[:-1] + f"; [그림 {num}])") if line.rstrip().endswith(")") else line + tag
        block = (f"\n\n![그림 {num}]({path})\n\n**[그림 {num}]** {caps[key]}")
        text = text[:text.rfind("\n", 0, i) + 1] + new_line + block + text[j:]
    return text


ROMAN = ["", "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ", "Ⅶ", "Ⅷ", "Ⅸ", "Ⅹ"]


def style_chapters(text):
    """장절 번호를 학교 양식으로 (2026-10-12). 졸업생 논문(2026-08): `Ⅰ. 서론` / `1. 연구 배경`.

    **본문 참조도 함께 바꾼다** — 제목만 로마자로 하면 「7.3절」이 가리키는 곳이 사라진다.
    `N.M절` → `Ⅹ장 M절`, `N장` → `Ⅹ장`.

    **「250장」과 충돌하지 않는다.** 장 참조는 한 자리이고 앞에 숫자·쉼표가 오지 않는다 —
    `(?<![0-9,])`가 「799장」의 `9장`, 「250장」의 `0장`을 막는다(0은 [1-9]가 아니다).
    「4·6장」처럼 묶어 쓴 것도 각 숫자를 바꾼다.

    장별 초안은 손대지 않는다 — 초안은 `# 4장 …` / `## 4.8 …`로 쓰고 여기서 양식을 입힌다.
    """
    def chap_title(m):
        return f"# {ROMAN[int(m.group(1))]}. {m.group(2)}"
    text = re.sub(r"^# (\d+)장 (.*)$", chap_title, text, flags=re.M)
    # 절 제목: 장 안에서 1부터 — `## 4.8 제목` → `## 8. 제목`
    text = re.sub(r"^## \d+\.(\d+) (.*)$", lambda m: f"## {m.group(1)}. {m.group(2)}", text, flags=re.M)
    # 절 참조: `7.3절` → `Ⅶ장 3절`
    text = re.sub(r"(?<![\d.])(\d+)\.(\d+)절",
                  lambda m: f"{ROMAN[int(m.group(1))]}장 {m.group(2)}절", text)
    # 장 참조: `6장` · `4·6장` · `4~6장`
    def chap_ref(m):
        body = m.group(1)
        return "".join(ROMAN[int(c)] if c.isdigit() else c for c in body) + "장"
    text = re.sub(r"(?<![0-9,])([1-9](?:[·~][1-9])*)장", chap_ref, text)
    return text


def toc(text):
    """차례 (2026-10-12 학교 양식). 졸업생 논문(2026-08)의 배치를 따른다.

      앞부분(국문 초록·Abstract·표 차례·그림 차례)이 먼저 붙어서 오고,
      **장마다 앞에 빈 줄 두 개**, 절은 **네 칸 들여쓰기**, 마지막에 참고 문헌·감사의 글.

    **점선 리더와 쪽 번호는 넣지 않는다.** 졸업생 논문은 `Ⅰ. 서 론 ·········· 1`처럼 점선으로
    쪽수를 잇는데, 쪽수는 조판(한글) 전에는 정해지지 않는다. 점선만 넣으면 **허공으로 가는
    줄**이 되고, 한글에서는 탭 리더로 자동 생성하는 것이 정석이라 오히려 지워야 할 것이 된다.
    마크다운 합본은 **구조만** 맞추고 리더는 조판에서 붙인다.

    목록 기호(`- `)도 쓰지 않는다 — 졸업생 논문의 차례는 평문이다.
    """
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^(#{1,2}) (.*)", ln)
        if not m:
            continue
        title = m.group(2).replace("`", "")
        if title.startswith("차 례"):
            continue
        if len(m.group(1)) == 2:
            out.append("    " + title)
            continue
        if re.match(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\.", title):   # 장 앞은 두 줄 띄운다
            out += ["", ""]
        out.append(title)
    return "\n".join(out)


def number_tables(text):
    """표 번호와 캡션 위치 (2026-10-12 학교 양식으로 개정).

    **전체 일련번호**(`[표 1]`~)이고 **캡션은 표 아래**다 — 졸업생 논문(2026-08)이 그렇다.
    2026-10-02 판은 장별 번호(`표 4-2`)에 캡션이 표 **위**였는데, 둘 다 양식과 달랐다.

    장별 초안은 여전히 `**표 [키].** 제목`을 표 **앞**에 적는다(쓰기 편한 자리다). 이 함수가
    캡션을 떼어 **뒤따르는 표 블록 아래로 옮긴다.** 본문의 `[표:키]`는 `[표 N]`으로 바꾼다.
    """
    lines = text.split("\n")
    out, numbers, items = [], {}, []
    k, i = 0, 0
    while i < len(lines):
        m = re.match(r"^\*\*표 \[([\w-]+)\]\.\*\* (.*)$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        key, title = m.group(1), m.group(2)
        if key in numbers:
            raise SystemExit(f"표 키 중복: {key}")
        k += 1
        numbers[key] = str(k)
        items.append(f"표 {k}. {title.replace('`', '')}")
        i += 1
        blanks = []
        while i < len(lines) and not lines[i].strip():
            blanks.append(lines[i])
            i += 1
        tbl = []
        while i < len(lines) and lines[i].lstrip().startswith("|"):
            tbl.append(lines[i])
            i += 1
        if not tbl:
            raise SystemExit(f"표 {key}의 캡션 뒤에 표가 없다 — 캡션은 표 바로 앞에 둔다")
        out.extend(blanks)
        out.extend(tbl)
        out.append("")
        out.append(f"**[표 {k}]** {title}")
    text = "\n".join(out)

    def ref(m):
        if m.group(1) not in numbers:
            raise SystemExit(f"없는 표를 가리킨다: [표:{m.group(1)}]")
        return f"[표 {numbers[m.group(1)]}]"
    text = re.sub(r"\[표:([\w-]+)\]", ref, text)
    return text, "\n".join(items)


def figure_list(text):
    """그림 목차 — 캡션 첫 문장을 제목으로 쓴다 (2026-10-02)."""
    items = []
    # 캡션은 `**[그림 N]** …` (학교 양식). `\d+`이므로 번호가 10을 넘어도 빠지지 않는다
    for m in re.finditer(r"^\*\*\[그림 (\d+)\]\*\* (.*)$", text, flags=re.M):
        first = re.split(r"(?<=\.)\s", m.group(2), maxsplit=1)[0].rstrip(".")
        items.append(f"그림 {m.group(1)}. {first.replace('**', '').replace('`', '')}")
    return "\n".join(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="thesis_full_draft.md")
    args = ap.parse_args()

    abstract = body_of("thesis_abstract_draft.md")
    ko = abstract[abstract.index("## 국문 초록"):abstract.index("\n---\n")].replace("## 국문 초록", "# 국문 초록")
    en = abstract[abstract.index("## Abstract"):].replace("## Abstract", "# Abstract")

    parts, dropped = [], 0
    for f in CHAPTERS:
        b, d = drop_correction_quotes(body_of(f))
        dropped += d
        parts.append(cite_chapter(b))
    main_text = "\n\n---\n\n".join(parts)

    # 인용 번호: 처음 나오는 순서
    order = []
    for m in re.finditer(r"«R(\d+)»", main_text):
        n = int(m.group(1))
        if n not in order:
            order.append(n)
    new_no = {old: i + 1 for i, old in enumerate(order)}
    main_text = re.sub(r"«R(\d+)»", lambda m: f"[{new_no[int(m.group(1))]}]", main_text)

    refs = parse_references()
    uncited = sorted(set(refs) - set(order))
    # `[N] 저자, "제목," 출처, 년.` — 졸업생 논문(2026-08)의 표기 (2026-10-12)
    bib = "\n".join(f"[{new_no[o]}] {refs[o]}" for o in order)

    main_text = insert_figures(main_text, parse_captions())
    main_text, table_items = number_tables(main_text)
    main_text = style_chapters(main_text)      # 장절 번호와 본문 참조를 함께 바꾼다

    today = dt.date.today().isoformat()
    head = (f"<!-- 자동 생성: scripts/build_thesis.py ({today}). 직접 고치지 말 것 — 장별 초안을 고치고 다시 생성한다. -->\n\n"
            "# 자율주행 비전-언어 모델의 저정밀 배포를 위한 표현 범위 여유 측정과 진단\n\n"
            "*Measuring and Diagnosing Representable-Range Headroom for Low-Precision Deployment of "
            "Autonomous-Driving Vision-Language Models*\n\n"
            f"석사학위논문 초안 · 합본 생성 {today} · 학교 양식 적용 전\n\n---\n\n")
    # 구성 순서 (2026-10-12 학교 양식) — 졸업생 논문(2026-08):
    #   표지 → 속표지 → 차례 → Abstract → 표 차례 → 그림 차례 → 본문 → 참고 문헌 → 감사의 글
    # **국문 초록은 남긴다 (사용자 결정)** — 졸업생 논문에는 없으나 빼야 할 이유도 없다.
    # 한국 논문 관행대로 Abstract 앞에 둔다. 표지·속표지·쪽 번호는 조판 단계(한글)에서 붙인다.
    ack = body_of("thesis_acknowledgment_draft.md")
    body = (f"{ko}\n\n---\n\n{en}\n\n---\n\n{main_text}"
            f"\n\n---\n\n# 참고 문헌\n\n{bib}\n\n---\n\n{ack}\n")
    toc_items, figure_items = toc(body), figure_list(main_text)
    # 표·그림 차례는 별도 쪽이라 본문 제목 훑기에 잡히지 않는다 — 차례에 손으로 끼운다
    toc_lines = toc_items.split("\n")
    for idx, ln in enumerate(toc_lines):
        if ln.strip() == "Abstract":
            toc_lines[idx + 1:idx + 1] = ["표 차례", "그림 차례"]
            break
    else:
        raise SystemExit("차례에 Abstract 항목이 없다 — 초록 제목을 확인한다")
    toc_items = "\n".join(toc_lines)
    front = (f"# 차 례\n\n{toc_items}\n\n---\n\n{ko}\n\n---\n\n{en}"
             f"\n\n---\n\n# 표 차례\n\n{table_items}\n\n---\n\n# 그림 차례\n\n{figure_items}")
    lists = front
    out = (head + front + "\n\n---\n\n" + main_text
           + f"\n\n---\n\n# 참고 문헌\n\n{bib}\n\n---\n\n{ack}\n")
    # 목차만 따로 (2026-10-02). 합본과 같은 실행에서 쓰므로 항상 맞는다
    toc_doc = head.replace("석사학위논문 초안 · 합본 생성", "석사학위논문 목차 · 합본 생성") + lists + "\n"

    # 원고 집계 (2026-10-02). **사람이 grep으로 세지 않는다** — `**표 3-3에서 주장하는…**` 같은
    # 문단 제목이 표 제목 패턴에 걸리고, `## 그림 목차`·`## 표 목차`가 본문 절로 세어진다.
    # 둘 다 실제로 틀렸다(표 41·절 59로 보고했으나 40·57이었다). 아래는 목차를 만든 바로 그
    # 자료에서 세므로 정의상 어긋날 수 없고, 어긋나면 그것이 결함이라 아래 검증이 멈춘다.
    # 차례의 절은 **네 칸 들여쓰기**다(2026-10-12 양식). 2026-10-02 판은 `  - `였고,
    # 양식을 바꾸면서 이 집계가 0을 내도 아무도 막지 않았다 — 그래서 아래에 대조를 넣었다.
    n_sec = sum(1 for l in toc_items.split("\n") if l.startswith("    ") and l.strip())
    n_tab = len([l for l in table_items.split("\n") if l.strip()])
    n_fig = len([l for l in figure_items.split("\n") if l.strip()])

    # 번호를 손으로 적은 표가 섞이면 본문과 목차가 어긋난다 — 번호 매김 뒤 패턴 수로 잡는다
    # 절 참조가 실재하는 절을 가리키는가 (2026-10-12). 절을 합치면서 참조 57건을 다시
    # 매겼는데, `thesis_figures.md`·초록·참고문헌을 **치환 대상에서 빠뜨려** 그림 캡션이
    # 사라진 절(`Ⅳ장 8절`)을 가리키고 있었다. 수동 확인으로 겨우 잡았으므로 여기 건다.
    chap_secs, cur = {}, None
    for ln in main_text.split("\n"):
        m = re.match(r"^# ([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])\. ", ln)
        if m:
            cur = m.group(1)
            chap_secs[cur] = 0
        elif cur and re.match(r"^## \d+\.", ln):
            chap_secs[cur] += 1
    bad = sorted({(c, n) for c, n in re.findall(r"([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])장 (\d+)절", main_text)
                  if int(n) > chap_secs.get(c, 0)})
    if bad:
        raise SystemExit("없는 절을 가리키는 참조: "
                         + ", ".join(f"{c}장 {n}절" for c, n in bad)
                         + f" — 장별 절 수는 {chap_secs}")

    n_body_sec = len(re.findall(r"^## ", main_text, flags=re.M))
    if n_body_sec != n_sec:
        raise SystemExit(f"절 수 불일치: 본문 {n_body_sec} ≠ 차례 {n_sec} — "
                         "차례의 들여쓰기 규칙과 집계가 어긋났는지 본다")

    n_body_tab = len(re.findall(r"^\*\*\[표 \d+\]\*\*", main_text, flags=re.M))
    if n_body_tab != n_tab:
        raise SystemExit(f"표 수 불일치: 본문 {n_body_tab} ≠ 목차 {n_tab} — "
                         "`**[표 N]**`를 직접 쓴 표가 있는지 본다 (키 형식 `**표 [키].**`를 쓸 것)")
    if n_fig != len(FIGURES):
        raise SystemExit(f"그림 수 불일치: 캡션 {n_fig} ≠ 삽입 설정 {len(FIGURES)} — "
                         "thesis_figures.md의 캡션과 FIGURES 목록 둘 다 본다")

    n_cite = len(re.findall(r"\[\d+\]", main_text))
    print(f"장 {len(CHAPTERS)}개 · 정정 블록 {dropped}개 제거 · 인용 표지 {n_cite}개 · 인용 문헌 {len(order)}개")
    print(f"절 {n_sec}개 · 표 {n_tab}개 · 그림 {n_fig}개 — 본문·목차 대조 통과")
    print(f"그림 {len(FIGURES)}개 · 번호 대응(옛→새): " + ", ".join(f"{o}→{new_no[o]}" for o in order))
    if uncited:
        print(f"⚠ 인용되지 않은 참고문헌(옛 번호): {uncited}")
    if not args.check:
        (ROOT / args.out).write_text(out)
        print(f"→ {args.out} ({out.count(chr(10)) + 1}줄)")
        toc_path = Path(args.out).with_name("thesis_toc.md")
        (ROOT / toc_path).write_text(toc_doc)
        print(f"→ {toc_path} ({toc_doc.count(chr(10)) + 1}줄)")


if __name__ == "__main__":
    main()
