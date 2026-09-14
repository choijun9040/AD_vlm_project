"""
학습 손실 곡선 시각화 (KSAE 2026 추계 원고용)
================================================
`logs/*.log`의 스텝 로그를 파싱해 두 종류의 그림을 만든다.

  1) ksae_fig_loss.png — 원고용 2-패널 비교도
     (a) 5개 변형의 L_task — 손실 항이 달라도 과제 손실 자체는 동등함을 보인다
     (b) 증류 항 곡선 — L_align / L_atc / L_kd
  2) <variant>_loss_curve.png — 변형별 상세 패널 (부록용)

명칭은 paper_positioning_ksae2026.md의 결정을 따른다: L_spatial→L_align, L_temporal→L_atc.
로그에 남은 키(spatial/temporal)는 파싱용이고 표시만 새 이름으로 바꾼다.
한글 폰트가 없는 환경이라 축·범례는 영문으로 둔다(KSAE 원고에서도 통용).

실행:  python scripts/plot_loss_curves.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("logs")
OUT_DIR = Path("loss_graph")

# (표시명, 로그 파일, 학습된 증류 항 — λ=0으로 죽은 항은 제외)
RUNS = [
    ("Full (L_task+L_align+L_atc)", "full.log",        ["align", "atc"]),
    ("L_align only",                "spatial.log",     ["align"]),
    ("L_atc only",                  "temporal.log",    ["atc"]),
    ("Baseline (no KD)",            "baseline_v2.log", []),
    ("KD-only (uniform)",           "kd_only_v4.log",  ["kd"]),
]

# 로그 키 → 표시 이름
DISPLAY = {"task": "L_task", "align": "L_align", "atc": "L_atc", "kd": "L_kd"}
COLOR   = {"task": "#333333", "align": "#0d6b59", "atc": "#a8371a", "kd": "#8a6a12"}

STEP_RE  = re.compile(r"step (\d+)/\d+ \| loss ([\d.]+)")
# 2026-09-14 로그 키 변경(spatial→align, temporal→atc). 이미 쌓인 로그는 구 키이므로
# 양쪽을 모두 받는다.
COMP_RE  = re.compile(r"(task|spatial|align|temporal|atc|kd)=([\d.]+)")


def parse(log_path: Path) -> dict:
    """스텝별 total/task/align/atc/kd 손실을 뽑는다."""
    out = {"step": [], "total": [], "task": [], "align": [], "atc": [], "kd": []}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = STEP_RE.search(line)
        if not m:
            continue
        out["step"].append(int(m.group(1)))
        out["total"].append(float(m.group(2)))
        comps = {k: float(v) for k, v in COMP_RE.findall(line)}
        # baseline은 분해 없이 total만 로깅되므로 total을 task로 취급한다
        out["task"].append(comps.get("task", float(m.group(2))))
        out["align"].append(comps.get("align", comps.get("spatial", float("nan"))))
        out["atc"].append(comps.get("atc", comps.get("temporal", float("nan"))))
        out["kd"].append(comps.get("kd", float("nan")))
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    data = {}
    for name, log, trained in RUNS:
        p = LOG_DIR / log
        if not p.exists():
            print(f"  건너뜀 (로그 없음): {log}")
            continue
        data[name] = (parse(p), trained)
        print(f"  {name:30s} {len(data[name][0]['step'])}점")

    # ---- 1) 원고용 2-패널 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    cmap = plt.get_cmap("tab10")

    for i, (name, (d, _)) in enumerate(data.items()):
        ax1.plot(d["step"], d["task"], lw=1.1, color=cmap(i), label=name)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("$L_{task}$")
    ax1.set_title("(a) Task loss is comparable across variants", fontsize=10.5)
    ax1.legend(fontsize=7.5, frameon=False)
    ax1.grid(alpha=0.25, lw=0.5)

    # L_kd가 L_align/L_atc보다 한 자릿수 커서 선형 축에서는 나머지가 눌린다 → log 축
    for name, (d, trained) in data.items():
        short = "Full" if name.startswith("Full") else name.replace(" only", "")
        for key in trained:
            ax2.plot(d["step"], d[key], lw=1.1, color=COLOR[key],
                     alpha=0.55 if "only" in name else 1.0,
                     ls="--" if "only" in name else "-",
                     label=f"{DISPLAY[key]} ({short})")
    ax2.set_yscale("log")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Distillation loss (log)")
    ax2.set_title("(b) Distillation terms — single-loss curves\noverlap the Full run", fontsize=10.5)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    out = OUT_DIR / "ksae_fig_loss.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"저장: {out}")

    # ---- 2) 변형별 상세 패널 ----
    for name, (d, trained) in data.items():
        keys = ["task"] + trained
        fig, axes = plt.subplots(1, len(keys) + 1, figsize=(3.1 * (len(keys) + 1), 3.0))
        axes[0].plot(d["step"], d["total"], lw=1.0, color="#14191b")
        axes[0].set_title("$L_{total}$", fontsize=10)
        for ax, key in zip(axes[1:], keys):
            ax.plot(d["step"], d[key], lw=1.0, color=COLOR[key])
            ax.set_title(f"${DISPLAY[key].replace('L_', 'L_{')}}}$", fontsize=10)
        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(alpha=0.25, lw=0.5)
        fig.suptitle(name, fontsize=10.5)
        fig.tight_layout()
        tag = name.split(" ")[0].lower().strip("(")
        out = OUT_DIR / f"{tag}_loss_curve.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
