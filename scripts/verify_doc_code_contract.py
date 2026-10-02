#!/usr/bin/env python3
"""문서의 논거 ↔ 코드 구현 대조 (원장 L9). 2026-10-02 신설.

**무엇을 잡는가.** 문서에서 판단이 바뀌었는데 그것을 구현한 코드가 따라오지 않는 부류.
2026-09-17에 두 번 나왔고(게이트 통계량, DLA의 bf16), **값 검사도 출처 검사도 통과한다**
— 문서 안에서는 각각 일관되기 때문이다. 어긋난 것은 문서와 산출물 사이다.

**어디까지 닫는가.** `docs/doc_code_contract.yaml`에 등록된 쌍까지다. 등록 밖은 사람 몫이다
— 그래서 원장 L9는 「절반 닫힘」이고, 이 검사기가 있다고 완전히 닫히지 않는다.

**docstring과 주석은 보지 않는다.** 설명이 아니라 동작을 봐야 한다. 실제로 `plan_fix`의
docstring에는 "p50"이 (과거 판을 설명하려고) 나오므로, 텍스트 검색으로는 오탐이 난다.
그래서 AST로 식별자·문자열 키만 모은다.

실행:
    python scripts/verify_doc_code_contract.py
    python scripts/verify_doc_code_contract.py --manifest docs/doc_code_contract_regress.yaml  # 회귀(반드시 1)

검사 유형 다섯: function+must_use / must_not_use / must_say(도구가 내는 말) ·
constant+value · argparse(기본값 자체가 논거인 자리).
"""

import argparse
import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def used_names(fn):
    """함수가 **실제로 쓰는** 이름 — docstring을 뺀 본문의 식별자·속성·문자열."""
    body = fn.body
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    names = set()
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.keyword) and n.arg:
                names.add(n.arg)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                names.add(n.value)
    return names


def said_strings(fn):
    """함수가 **내는 말** — docstring을 뺀 본문의 문자열 리터럴 전부.

    `must_say`는 부분 일치로 본다. 출력 문장은 괄호로 이어 붙여 조각나 있고(f-string도
    조각난다), 문구를 글자 그대로 고정하면 사소한 수정마다 검사가 깨져 **아무도 믿지 않는
    검사**가 된다. 보려는 것은 "도구가 이 말을 하는가"이지 문장의 형태가 아니다.
    """
    body = fn.body
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    out = []
    for stmt in body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
    return out


def argparse_defaults(tree):
    """`ap.add_argument("--opt", ..., default=X)` → {"--opt": X}.

    문서가 *"주지 않으면 모델 자신의 설정을 쓴다"*처럼 **기본값 자체를 근거로 삼는** 자리가
    있다. 그 자리에 값이 박히면 문서가 거짓이 되는데, 식별자 검사로는 보이지 않는다.
    """
    out = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str)):
            opt = n.args[0].value
            for kw in n.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    out[opt] = kw.value.value
                elif kw.arg == "default":
                    out[opt] = "<non-literal>"
            out.setdefault(opt, "<absent>")
    return out


def module_constants(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant):
                    out[t.id] = node.value.value
    return out


def check_code(spec):
    path = ROOT / spec["file"]
    if not path.exists():
        return [f"코드 파일 없음: {spec['file']}"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems = []

    if "constant" in spec:
        consts = module_constants(tree)
        name = spec["constant"]
        if name not in consts:
            problems.append(f"모듈 상수 없음: {name} ({spec['file']})")
        elif consts[name] != spec["value"]:
            problems.append(f"상수 불일치: {name} = {consts[name]!r}, 선언은 {spec['value']!r}")

    if "argparse" in spec:
        defaults = argparse_defaults(tree)
        for item in spec["argparse"]:
            opt = item["option"]
            if opt not in defaults:
                problems.append(f"인자 없음: {opt} ({spec['file']})")
            elif defaults[opt] != item["default"]:
                problems.append(f"기본값 불일치: {opt} = {defaults[opt]!r}, 선언은 {item['default']!r}"
                                " — 문서가 기본값을 근거로 삼는 자리다")

    if "function" in spec:
        fns = {n.name: n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        fname = spec["function"]
        if fname not in fns:
            problems.append(f"함수 없음: {fname}() ({spec['file']})")
        else:
            names = used_names(fns[fname])
            for want in spec.get("must_use", []):
                if want not in names:
                    problems.append(f"{fname}()가 «{want}»를 쓰지 않는다 — 문서의 논거가 코드에 없다")
            for bad in spec.get("must_not_use", []):
                if bad in names:
                    problems.append(f"{fname}()가 «{bad}»를 쓴다 — 문서가 쓰지 않기로 한 것이다")
            said = said_strings(fns[fname])
            for phrase in spec.get("must_say", []):
                if not any(phrase in s for s in said):
                    problems.append(f"{fname}()가 «{phrase}»를 말하지 않는다 — "
                                    "문서는 도구가 이것을 알린다고 적고 있다")
    return problems


def check_docs(pair):
    problems = []
    for f in pair.get("docs", []):
        path = ROOT / f
        if not path.exists():
            problems.append(f"문서 없음: {f}")
            continue
        text = path.read_text(encoding="utf-8")
        for want in pair.get("docs_must_contain", []):
            if want not in text:
                problems.append(f"{f}에 «{want}»가 없다 — 코드는 있는데 문서가 주장을 잃었다")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--manifest", default="docs/doc_code_contract.yaml")
    args = ap.parse_args()

    spec = yaml.safe_load((ROOT / args.manifest).read_text(encoding="utf-8"))
    pairs = spec["pairs"]
    print(f"문서↔코드 계약 {len(pairs)}쌍\n")

    failed = 0
    for pair in pairs:
        problems = check_code(pair["code"]) + check_docs(pair)
        mark = "✗" if problems else "✓"
        print(f"{mark} [{pair['id']}] {pair['claim']}")
        if problems:
            failed += 1
            for p in problems:
                print(f"      ✗ {p}")
            if pair.get("broke_once"):
                print(f"      ↳ 전례: {pair['broke_once']}")
        print()

    print("─" * 72)
    if failed:
        print(f"✗ {failed}쌍이 어긋났다. **문서와 코드 중 어느 쪽이 맞는지 정하고 양쪽을 맞춘다.**")
        print("  이 부류는 값 검사·출처 검사가 못 잡는다 — 문서 안에서는 일관되기 때문이다.")
        return 1
    print(f"문제 없음 — {len(pairs)}쌍 모두 문서의 논거가 코드에 있다.")
    print("등록되지 않은 쌍은 여전히 사람이 본다 (원장 L9는 「절반 닫힘」).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
