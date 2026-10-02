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
  5. 목차·그림 목차·표 목차를 만든다(그림 제목은 캡션 첫 문장, 표는 `**표 [키].** 제목`에 장별 번호, 본문 `[표:키]`는 번호로).

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

# ── 그림: 번호 → (파일, 기준 줄이 포함하는 글자) ─────────────────────────────────────────
FIGURES = [
    ("4-1", "figures/fig1_activation_profile.png", "(799장, 평가 해상도, bf16, 이미지별 최대의 중앙값)"),
    ("4-2", "figures/fig2_resolution_collapse.png", "(fp16 붕괴율, 250장, 이미지 단위로 출력이 NaN이면 붕괴)"),
    ("4-3", "figures/fig4_metric_inversion.png", "(원본 해상도, 250장, 마지막 블록, 여유는 p50·bf16 측정)"),
    ("4-4", "figures/fig3_lambda_dose_response.png", "(799장, 평가 해상도, bf16, p50)"),
    ("6-1", "figures/fig6_runtime_collapse.png", "(250장, 같은 순서, 15W 모드, TensorRT 10.3, 강타입. 지연은 두 분할 엔진의 합)"),
    ("7-1", "figures/fig_gate_calibration.png", "최악 여유 1.00에서 붕괴 2%, 1.05에서 0%다. 따로 보정할 상수가 없다."),
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
    for num, path, anchor in FIGURES:
        i = text.find(anchor)
        if i < 0:
            raise SystemExit(f"그림 {num}의 기준 줄을 찾지 못했다: {anchor[:40]}")
        j = text.index("\n", i)
        line = text[text.rfind("\n", 0, i) + 1:j]
        tag = f" (그림 {num})" if not line.rstrip().endswith(")") else ""
        new_line = (line[:-1] + f"; 그림 {num})") if line.rstrip().endswith(")") else line + tag
        block = (f"\n\n![그림 {num}]({path})\n\n**그림 {num}.** {caps[num]}")
        text = text[:text.rfind("\n", 0, i) + 1] + new_line + block + text[j:]
    return text


def toc(text):
    items = []
    for ln in text.split("\n"):
        m = re.match(r"^(#{1,2}) (.*)", ln)
        if m and not m.group(2).startswith("목차"):
            items.append(("  " if len(m.group(1)) == 2 else "") + "- " + m.group(2).replace("`", ""))
    return "\n".join(items)


def number_tables(text):
    """표 번호 (2026-10-02). 장별 초안의 `**표 [키].** 제목`에 장별 번호(표 4-2)를 매기고, 본문의
    `[표:키]`를 그 번호로 바꾼다. 번호는 장 안에서 나오는 순서다 — 표를 넣거나 빼도 다시 맞춰진다."""
    out, numbers, items, chap, k = [], {}, [], None, 0
    for ln in text.split("\n"):
        m = re.match(r"^# (\d+)장", ln)
        if m:
            chap, k = int(m.group(1)), 0
        m = re.match(r"^\*\*표 \[([\w-]+)\]\.\*\* (.*)$", ln)
        if m:
            if chap is None:
                raise SystemExit(f"장 밖의 표: {m.group(1)}")
            if m.group(1) in numbers:
                raise SystemExit(f"표 키 중복: {m.group(1)}")
            k += 1
            no = f"{chap}-{k}"
            numbers[m.group(1)] = no
            items.append(f"- 표 {no}. {m.group(2).replace('`', '')}")
            ln = f"**표 {no}.** {m.group(2)}"
        out.append(ln)
    text = "\n".join(out)

    def ref(m):
        if m.group(1) not in numbers:
            raise SystemExit(f"없는 표를 가리킨다: [표:{m.group(1)}]")
        return f"표 {numbers[m.group(1)]}"
    text = re.sub(r"\[표:([\w-]+)\]", ref, text)
    return text, "\n".join(items)


def figure_list(text):
    """그림 목차 — 캡션 첫 문장을 제목으로 쓴다 (2026-10-02)."""
    items = []
    for m in re.finditer(r"^\*\*그림 (\d-\d)\.\*\* (.*)$", text, flags=re.M):
        first = re.split(r"(?<=\.)\s", m.group(2), maxsplit=1)[0].rstrip(".")
        items.append(f"- 그림 {m.group(1)}. {first.replace('**', '').replace('`', '')}")
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
    bib = "\n".join(f"{new_no[o]}. {refs[o]}" for o in order)

    main_text = insert_figures(main_text, parse_captions())
    main_text, table_items = number_tables(main_text)

    today = dt.date.today().isoformat()
    head = (f"<!-- 자동 생성: scripts/build_thesis.py ({today}). 직접 고치지 말 것 — 장별 초안을 고치고 다시 생성한다. -->\n\n"
            "# 자율주행 비전-언어 모델의 저정밀 배포를 위한 표현 범위 여유 측정과 진단\n\n"
            "*Measuring and Diagnosing Representable-Range Headroom for Low-Precision Deployment of "
            "Autonomous-Driving Vision-Language Models*\n\n"
            f"석사학위논문 초안 · 합본 생성 {today} · 학교 양식 적용 전\n\n---\n\n")
    body = f"{ko}\n\n---\n\n{main_text}\n\n---\n\n# 참고문헌\n\n{bib}\n\n---\n\n{en}\n"
    out = (head + "# 목차\n\n" + toc(body) + "\n\n## 그림 목차\n\n" + figure_list(main_text)
           + "\n\n## 표 목차\n\n" + table_items
           + "\n\n---\n\n" + body)

    n_cite = len(re.findall(r"\[\d+\]", main_text))
    print(f"장 {len(CHAPTERS)}개 · 정정 블록 {dropped}개 제거 · 인용 표지 {n_cite}개 · 인용 문헌 {len(order)}개")
    print(f"그림 {len(FIGURES)}개 · 번호 대응(옛→새): " + ", ".join(f"{o}→{new_no[o]}" for o in order))
    if uncited:
        print(f"⚠ 인용되지 않은 참고문헌(옛 번호): {uncited}")
    if not args.check:
        (ROOT / args.out).write_text(out)
        print(f"→ {args.out} ({out.count(chr(10)) + 1}줄)")


if __name__ == "__main__":
    main()
