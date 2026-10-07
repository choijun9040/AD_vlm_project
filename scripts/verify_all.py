#!/usr/bin/env python3
"""검사 전부를 한 번에 돌린다. 2026-10-02 신설.

**왜 필요한가.** 검사가 일곱으로 늘었고 **두 개는 종료코드 1이 정상**이다(검사기 자신의
회귀 시험). 사람이 그 반전을 기억해야 하면 언젠가 틀린다 — 회귀가 0을 내는데 통과로 읽거나,
1을 내는데 실패로 읽는다. 여기서는 기대 종료코드를 표에 적고 **스크립트가 대조한다.**

**`watch` 경고는 실패가 아니다.** 값이 아니라 조건 병기를 사람이 확인하는 부류(C8)다. 다만
**건수가 늘면 새로 생긴 것이므로** 기준선과 비교해 알린다. 줄어도 알린다 — 줄어든 것이
"고쳤다"로 읽히면 안 된다(개요 §7).

이름이 `check_all.py`가 아닌 이유: `.gitignore`가 `scripts/check*.py`를 지운다.

실행:
    python scripts/verify_all.py
    python scripts/verify_all.py --skip-smoke   # --help 전수 훑기를 건너뛴다 (느릴 때)
"""

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (이름, 명령, 기대 종료코드, 비고, 필요한 모듈)
# **필요한 모듈이 없으면 건너뛴다 — 실패가 아니다.** 문서·메타 검사 여섯은 pyyaml만 있으면
# 돌고, 도구 스모크만 torch가 필요하다. CI는 가벼운 환경에서 여섯을 돌리고 하나를 건너뛴다.
CHECKS = [
    ("수치 감사 (C1·C7·C8)", ["scripts/audit_numbers.py"], 0, "watch 건수는 아래에서 따로 본다", ()),
    # `--strict` 없이는 문제를 찍고도 **종료코드 0**이다. 원장 61행이 그 플래그를 빼고
    # "반드시 종료코드 1"이라 적어, 회귀 시험이 **늘 통과로 읽히는 상태**였다 (2026-10-02 발견).
    ("출처 대조 (C1~C3)", ["scripts/verify_claim_provenance.py", "--strict"], 0, "", ("yaml",)),
    ("출처 대조 — 회귀", ["scripts/verify_claim_provenance.py", "--strict",
                          "--manifest", "docs/claim_provenance_regress.yaml"], 1,
     "**1이 정상** — 합성 위반을 잡아야 한다", ("yaml",)),
    ("문서↔코드 계약 (C9)", ["scripts/verify_doc_code_contract.py"], 0, "", ("yaml",)),
    ("문서↔코드 계약 — 회귀", ["scripts/verify_doc_code_contract.py",
                               "--manifest", "docs/doc_code_contract_regress.yaml"], 1,
     "**1이 정상** — 합성 위반을 잡아야 한다", ("yaml",)),
    ("원장 정합성 (C10)", ["scripts/ledger_stats.py", "--quiet"], 0, "", ()),
    ("합본 집계·목차 대조", ["scripts/build_thesis.py", "--check"], 0, "", ()),
    ("도구 판정 분기 (O10)", ["scripts/headroom_guard.py", "selftest"], 0, "모델 없이 돈다", ("torch",)),
]

WATCH_BASELINE = 39     # 2026-10-07. 바꿀 때는 개요 §7에 사유를 적는다
#   33 → 39: 「완전 역전의 우연 확률」을 대장에 등록했다. ρ = −1.000이 6모델과 5변형 모두에서
#   성립해 우연 확률만 1/720·1/120으로 갈리므로, **n을 병기하지 않으면 두 값이 서로 틀린
#   것처럼 읽힌다.** 늘어난 6건은 모두 n이 병기된 자리다(확인함).


def have(mods):
    return [m for m in mods if importlib.util.find_spec(m) is None]


def run(cmd):
    p = subprocess.run([sys.executable] + cmd, cwd=ROOT,
                       capture_output=True, text=True, timeout=900)
    return p.returncode, p.stdout + p.stderr


def smoke():
    """argparse를 쓰는 스크립트 전부에 `--help` (원장 O10의 전수 훑기)."""
    targets = sorted(p for p in (ROOT / "scripts").glob("*.py")
                     if "argparse" in p.read_text(encoding="utf-8"))
    bad, skipped_smoke = [], []
    for p in targets:
        code, out = run([f"scripts/{p.name}", "--help"])
        if code == 0:
            continue
        # **의존성 부재는 argparse 결함이 아니다.** 스모크의 목적은 플래그 오타·설정 오류를
        # 잡는 것이고(원장 O10), CI의 가벼운 환경에서는 torch·transformers가 없다.
        if "ModuleNotFoundError" in out or "ImportError" in out:
            skipped_smoke.append(p.name)
            continue
        bad.append((p.name, out.strip().split("\n")[-1][:90]))
    return len(targets), bad, skipped_smoke


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("--skip-smoke", action="store_true", help="--help 전수 훑기를 건너뛴다")
    args = ap.parse_args()

    print(f"검사 {len(CHECKS)}개" + ("" if args.skip_smoke else " + --help 전수 훑기") + "\n")
    failed, notes, skipped = [], [], []

    for name, cmd, want, note, requires in CHECKS:
        absent = have(requires)
        if absent:
            print(f"  ⊘ {name:26s} 건너뜀 — {', '.join(absent)} 없음 (실패가 아니다)")
            skipped.append(name)
            continue
        code, out = run(cmd)
        ok = code == want
        print(f"  {'✓' if ok else '✗'} {name:26s} 종료 {code}" + (f"  ({note})" if note else ""))
        if not ok:
            failed.append((name, f"종료코드 {code}, 기대 {want}"))
            tail = [l for l in out.strip().split("\n") if l.strip()][-3:]
            for l in tail:
                print(f"        {l[:100]}")
        if name.startswith("수치 감사"):
            m = re.search(r"본문에 (\d+)건", out)
            if m:
                n = int(m.group(1))
                if n != WATCH_BASELINE:
                    notes.append(f"watch 경고 {n}건 — 기준선 {WATCH_BASELINE}과 다르다. "
                                 + ("늘었다면 새로 생긴 것이니 조건 병기를 확인하고, "
                                    if n > WATCH_BASELINE else "줄었다면 왜 줄었는지 적고, ")
                                 + "WATCH_BASELINE과 개요 §7을 함께 고친다")
                else:
                    print(f"        watch {n}건 — 기준선과 같다")

    if not args.skip_smoke:
        n, bad, sk = smoke()
        tail = f"{n}개" + (f" ({len(sk)}개는 의존성 없어 건너뜀)" if sk else "")
        print(f"  {'✓' if not bad else '✗'} {'--help 전수 훑기':24s} {tail}")
        for fname, msg in bad:
            print(f"        ✗ {fname}: {msg}")
        if bad:
            failed.append(("--help 스모크", f"{len(bad)}개 실패"))

    print("\n" + "─" * 72)
    for n in notes:
        print(f"△ {n}")
    if failed:
        print(f"✗ {len(failed)}개 검사가 실패했다")
        for name, why in failed:
            print(f"    ✗ {name} — {why}")
        return 1
    if skipped:
        print(f"⊘ {len(skipped)}개 건너뜀 (의존성 없음): {', '.join(skipped)}")
    print("✓ 통과" + (" (watch 경고는 조건 병기 확인용이며 실패가 아니다)" if not notes else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
