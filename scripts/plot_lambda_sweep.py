"""
그림 3 — 정렬 손실 가중치(λ_align)에 대한 용량-반응
======================================================
`_orchestrate_lambda_sweep.sh`가 남긴 `eval_results/lambda_sweep_activation_profile.json`을
읽는다. 네 지점(λ = 0 / 0.5 / 1.0 / 2.0) 모두 시드 1234·학습률 궤적 고정·5,000스텝으로
학습해 λ만 다르다.

두 패널로 나눈다.
  (a) 계층별 활성 크기 — **층마다 거동이 다르다**는 것이 요점이다.
      최종 블록은 단조 감소하는 반면 중간 계층은 λ=1.0에서 포화한다.
  (b) 최종 블록의 fp16 표현 범위 여유 — 배포에서 실제로 문제가 되는 양.
      1.0배 아래가 붕괴 구간이다.

붕괴율은 네 지점 모두 0%다(5,000스텝 모델은 활성이 한계 아래). 그래서 결과 변수를
붕괴율이 아니라 **연속량인 활성 크기**로 잡았다 — 붕괴율로 쟀다면 아무것도 보이지 않는다.

실행:
    python scripts/plot_lambda_sweep.py --layout double
    python scripts/plot_lambda_sweep.py --layout single
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP16_MAX = 65504.0

# (블록 index, 범례, 색, 선 스타일, 굵기, 마커)
SERIES = [
    (31, "block 31 (final)", "#0d6b59", "-",  2.6, "o"),
    (28, "block 28",         "#8a97a8", "-",  1.4, "^"),
    (24, "block 24",         "#a99b6f", "--", 1.4, "v"),
    (16, "block 16",         "#6b7674", ":",  1.4, "s"),
]

LAYOUT = {
    "double": dict(figsize=(7.4, 3.5), base_fs=10, tick_fs=9, legend_fs=8.5,
                   lw=1.0, ms=5.0, ncol=4),
    "single": dict(figsize=(3.4, 4.6), base_fs=7.5, tick_fs=6.5, legend_fs=6.4,
                   lw=0.62, ms=3.2, ncol=2),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="eval_results/lambda_sweep_activation_profile.json")
    ap.add_argument("--out", default="figures/fig3_lambda_dose_response")
    ap.add_argument("--layout", default="double", choices=list(LAYOUT))
    args = ap.parse_args()

    L = LAYOUT[args.layout]
    if args.layout != "double" and args.out == ap.get_default("out"):
        args.out = f"{args.out}_{args.layout}"

    prof = json.loads(Path(args.profile).read_text())
    pts = sorted((float(k.replace("sweep_align", "")), v)
                 for k, v in prof.items() if k != "_meta")
    lams = [p[0] for p in pts]
    n_img = prof.get("_meta", {}).get("n_images", "?")

    def val(entry, blk):
        return entry["passes"]["bfloat16"]["by_layer"][str(blk)]["p50"]

    plt.rcParams.update({"font.size": L["base_fs"], "axes.labelsize": L["base_fs"],
                         "xtick.labelsize": L["tick_fs"], "ytick.labelsize": L["tick_fs"]})
    nrows, ncols = (2, 1) if args.layout == "single" else (1, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=L["figsize"])
    ax_a, ax_b = axes.ravel()

    # --- (a) 계층별 활성 크기 ---
    for blk, label, color, ls, lw, mk in SERIES:
        ys = [val(e, blk) for _, e in pts]
        ax_a.plot(lams, ys, color=color, ls=ls, lw=lw * L["lw"], marker=mk,
                  ms=L["ms"], label=label, zorder=3)
    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"max$|$activation$|$  (median)")
    ax_a.set_title("(a) activation by depth", fontsize=L["base_fs"], pad=6)
    # 최종 블록만 단조라는 점을 그림 안에서 짚는다
    y31 = [val(e, 31) for _, e in pts]
    ax_a.annotate("monotonic", xy=(lams[-1], y31[-1]), xytext=(-4, 10),
                  textcoords="offset points", ha="right", fontsize=L["legend_fs"],
                  color="#0d6b59", fontweight="bold")
    y24 = [val(e, 24) for _, e in pts]
    ax_a.annotate("saturates", xy=(lams[-1], y24[-1]), xytext=(-4, -14),
                  textcoords="offset points", ha="right", fontsize=L["legend_fs"],
                  color="#8a6a12")

    # --- (b) fp16 여유 ---
    head = [FP16_MAX / val(e, 31) for _, e in pts]
    ax_b.axhspan(0, 1.0, color="#b03a1f", alpha=0.10, zorder=0)
    ax_b.axhline(1.0, color="#b03a1f", lw=1.2 * L["lw"], ls=(0, (5, 3)), zorder=1)
    ax_b.text(lams[-1], 1.06, "collapse below 1.0", fontsize=L["legend_fs"],
              color="#b03a1f", ha="right", va="bottom")
    ax_b.plot(lams, head, color="#0d6b59", lw=2.6 * L["lw"], marker="o",
              ms=L["ms"], zorder=3)
    for x, y in zip(lams, head):
        ax_b.annotate(f"{y:.2f}x", xy=(x, y), xytext=(0, 7), textcoords="offset points",
                      ha="center", fontsize=L["legend_fs"], color="#0d6b59")
    ax_b.set_ylabel("fp16 representable-range headroom")
    ax_b.set_ylim(0, max(head) * 1.28)
    ax_b.set_title("(b) deployment headroom", fontsize=L["base_fs"], pad=6)

    for ax in (ax_a, ax_b):
        ax.set_xlabel(r"$\lambda_{align}$")
        ax.set_xticks(lams)
        ax.set_xticklabels([f"{v:g}" for v in lams])
        ax.set_xlim(-0.12, max(lams) + 0.12)
        ax.grid(True, which="major", axis="y", color="#e4e8e4", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.legend(*ax_a.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.5, -0.02 if args.layout == "double" else -0.04),
               ncol=L["ncol"], fontsize=L["legend_fs"], frameon=False,
               handlelength=2.2, columnspacing=1.4)

    fig.tight_layout(rect=(0, 0.10 if args.layout == "double" else 0.07, 1, 1))
    for ext in ("png", "pdf"):
        out = Path(f"{args.out}.{ext}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"저장: {out}")

    print(f"이미지 {n_img}장 기준, λ = {lams}")
    print("  blk31: " + "  ".join(f"{v:,.0f}" for v in y31))
    print("  여유  : " + "  ".join(f"{v:.2f}x" for v in head))


if __name__ == "__main__":
    main()
