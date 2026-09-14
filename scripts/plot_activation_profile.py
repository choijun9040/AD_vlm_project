"""
그림 1 — 비전 인코더 계층별 활성 크기 프로파일과 fp16 표현 한계
=================================================================
`scripts/profile_vision_activations.py`가 남긴
`eval_results/vision_activation_profile.json`을 읽어 학술대회 논문용 그림을 만든다.

읽는 값은 bf16 통과(표현 범위가 넉넉해 NaN 없이 참값을 얻는 쪽)의 층별
max|activation| 중앙값이며, 이미지 799장(DriveLM val CAM_FRONT) 기준이다.

라벨은 KSAE 논문집 관례에 따라 영문으로 둔다(한글 폰트 의존도 제거 목적도 있다).

실행:
    python scripts/plot_activation_profile.py
    python scripts/plot_activation_profile.py --show-band false
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP16_MAX = 65504.0
ALIGN_LAYER = 23          # L_align이 실제로 정렬하는 블록 (CONFIG의 spatial_layer_idx)

# (JSON 키, 2단용 범례 표기, 1단용 축약 표기, 색, 선 스타일, 선 굵기, 밴드 표시 여부)
SERIES = [
    ("student_baseline_v2", "Task CE only",       "Task only",         "#b03a1f", "-",  2.4, True),
    ("student_temporal",    "Task + $L_{atc}$",   "Task + $L_{atc}$",  "#8a97a8", "-",  1.3, False),
    ("base_3b",             "Pre-trained (no FT)", "Pre-trained",      "#6b7674", "--", 1.3, False),
    ("student_kd_only_v4",  "Task + output KD",   "Task + out.KD",     "#a99b6f", "-",  1.3, False),
    ("teacher_7b",          "Teacher 7B",         "Teacher 7B",        "#2f4858", "-.", 1.3, False),
    ("student_spatial",     "Task + $L_{align}$", "Task + $L_{align}$", "#1f8a70", "--", 1.8, False),
    ("student_full",        "Task + $L_{align}$ + $L_{atc}$",
                            "Task + $L_{align}$ + $L_{atc}$",          "#0d6b59", "-",  2.4, True),
]

# 레이아웃별 치수 — KSAE 논문집은 2단 조판이라 단칼럼 폭이 약 8.5cm(3.35in)다.
LAYOUT = {
    "double": dict(figsize=(7.2, 4.4), base_fs=10, tick_fs=9, legend_fs=8.5,
                   lw_scale=1.0, legend_below=False, show_align_text=True),
    "single": dict(figsize=(3.4, 3.5), base_fs=7.5, tick_fs=6.5, legend_fs=6.2,
                   lw_scale=0.62, legend_below=True, show_align_text=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="eval_results/vision_activation_profile.json")
    ap.add_argument("--out", default="figures/fig1_activation_profile")
    ap.add_argument("--show-band", default="true", choices=["true", "false"],
                    help="주요 두 모델에 p50~p95 띠를 함께 표시할지")
    ap.add_argument("--layout", default="double", choices=list(LAYOUT),
                    help="double=2단 폭(17cm), single=단칼럼 폭(8.5cm)")
    args = ap.parse_args()

    L = LAYOUT[args.layout]
    if args.layout != "double" and args.out == ap.get_default("out"):
        args.out = f"{args.out}_{args.layout}"

    prof = json.loads(Path(args.profile).read_text())
    n_images = prof.get("_meta", {}).get("n_images", "?")
    band = args.show_band == "true"

    plt.rcParams.update({"font.size": L["base_fs"],
                         "axes.labelsize": L["base_fs"],
                         "xtick.labelsize": L["tick_fs"],
                         "ytick.labelsize": L["tick_fs"]})
    fig, ax = plt.subplots(figsize=L["figsize"])

    # L_align이 정렬하는 계층 표시 — 효과가 그 층에만 국한되지 않음을 함께 보이기 위함
    ax.axvline(ALIGN_LAYER, color="#c9ccc8", lw=1.0, zorder=0)
    if L["show_align_text"]:
        ax.text(ALIGN_LAYER - 0.55, 11.5, f"aligned block ({ALIGN_LAYER})",
                rotation=90, va="bottom", ha="right", fontsize=7.5, color="#8a918c")

    ax.axhline(FP16_MAX, color="#b03a1f", lw=1.3 * L["lw_scale"], ls=(0, (5, 3)), zorder=1)
    ax.text(0.3, FP16_MAX * 1.2,
            "fp16 limit (65,504)" if args.layout == "single"
            else "fp16 representable limit (65,504)",
            fontsize=L["legend_fs"] + 0.5, color="#b03a1f", va="bottom")

    for key, label_d, label_s, color, ls, lw, do_band in SERIES:
        if key not in prof:
            continue
        by_layer = prof[key]["passes"]["bfloat16"]["by_layer"]
        idx = sorted(int(i) for i in by_layer)
        p50 = [by_layer[str(i)]["p50"] for i in idx]
        ax.plot(idx, p50, color=color, ls=ls, lw=lw * L["lw_scale"],
                label=(label_s if args.layout == "single" else label_d), zorder=3)
        if band and do_band:
            p95 = [by_layer[str(i)]["p95"] for i in idx]
            ax.fill_between(idx, p50, p95, color=color, alpha=0.13, lw=0, zorder=2)

    # 헤드라인 수치: 정렬 손실 유무에 따른 중간 계층 격차
    a = prof["student_baseline_v2"]["passes"]["bfloat16"]["by_layer"]["24"]["p50"]
    b = prof["student_full"]["passes"]["bfloat16"]["by_layer"]["24"]["p50"]
    ax.annotate("", xy=(24, a), xytext=(24, b),
                arrowprops=dict(arrowstyle="<->", color="#3c4340",
                                lw=1.2 * L["lw_scale"]), zorder=4)
    ax.text(24.8, (a * b) ** 0.5, f"{a / b:.0f}$\\times$",
            fontsize=L["base_fs"], color="#3c4340", va="center", fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlim(-0.6, 31.6)
    ax.set_ylim(10, 1.3e5)
    ax.set_xlabel("Vision encoder block index")
    ax.set_ylabel(r"max$|$activation$|$  (median over images)")
    ax.set_xticks(range(0, 32, 4) if args.layout == "double" else range(0, 32, 8))
    ax.grid(True, which="major", axis="y", color="#e4e8e4", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if L["legend_below"]:
        # 단칼럼에서는 7개 항목을 그림 안에 넣으면 곡선을 가리므로 축 아래 2열로 뺀다
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), fontsize=L["legend_fs"],
                  frameon=False, ncol=2, handlelength=1.9, labelspacing=0.28,
                  columnspacing=1.0, borderaxespad=0.0)
    else:
        # 2단 폭에서는 좌측 중상단 여백에 둔다 — 곡선은 오른쪽에서 올라오고,
        # 맨 위는 fp16 한계 라벨이 차지하므로 그 아래로 내린다
        ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.88), fontsize=L["legend_fs"],
                  frameon=False, ncol=1, handlelength=2.4, labelspacing=0.4)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = Path(f"{args.out}.{ext}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"저장: {out}")

    print(f"이미지 {n_images}장 기준, blk24 격차 {a:,.0f} vs {b:,.0f} = {a / b:.1f}배")


if __name__ == "__main__":
    main()
