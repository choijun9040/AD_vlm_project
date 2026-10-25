"""
관련 연구 그림 — 선행 연구의 위치와 공백 (학위 논문 2.4절, 그림 2-1).
  행: 무엇으로 평가하는가 (과제 정확도 / 상대 분포: 이상치 비율·양자화 SNR / 절대 크기: 최대 활성·fp16 오버플로)
  열: 무엇을 비교하는가 (한 사전학습 모델·아키텍처의 성질 / 학습·미세조정 설계에 따른 차이)
  오른쪽 아래 칸(절대 크기 × 학습 설계에 따른 차이)이 2.4절의 선행 연구 공백이며 본 연구의 위치다.
데이터: 측정 결과가 아니라 2.1·2.2절의 서술을 옮긴 것이다. 연구를 더하거나 옮기면 2장 본문과 함께 고친다.
실행:
    python scripts/plot_related_map.py                 # 2단 폭
    python scripts/plot_related_map.py --layout single # 1단 폭
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK, INK2, GRID = "#0b0b0b", "#52514e", "#c9c9c5"
HL, HL_FC = "#eb6834", "#fde8de"

ROWS = ["Task accuracy",
        "Relative distribution\n(outlier ratio,\nquantization SNR)",
        "Absolute magnitude\n(max activation,\nfp16 overflow)"]
COLS = ["Property of one pre-trained\nmodel or architecture",
        "Difference across training\nor fine-tuning designs"]
# (행, 열) → 연구. 2.1절: 증류 계열은 정확도로 평가. 2.2절: 양자화 친화 학습·이상치 완화는 상대 분포,
# 대규모 활성·fp16 오버플로 보고는 한 모델·아키텍처의 성질로서의 절대 크기.
CELLS = {
    (0, 0): [],
    (0, 1): ["Drive-KD", "SCPNet", "EM-VLM4AD", "MiniDrive"],
    (1, 0): ["RegCache", "component-wise VLM\nquantization (2607.08029)"],
    (1, 1): ["CrAFT", "Colinearity Decay", "RoLoRA · RoSTE", "QuantTune", "Edge AI for VRU Safety"],
    (2, 0): ["Massive Activations", "Measuring Max. Activations", "PASA", "vLLM issue #40290"],
    (2, 1): None,   # 공백 — 본 연구
}

LAYOUT = {
    "double": dict(figsize=(7.0, 3.6), base_fs=8.5, small_fs=7.4, head_w=0.24),
    "single": dict(figsize=(3.5, 4.6), base_fs=7.0, small_fs=6.0, head_w=0.30),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="double", choices=list(LAYOUT))
    ap.add_argument("--out", default="figures/fig0b_related_map")
    args = ap.parse_args()
    L = LAYOUT[args.layout]

    fig, ax = plt.subplots(figsize=L["figsize"])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    hw, hh = L["head_w"], 0.16                      # 행 머리 폭, 열 머리 높이
    cw, ch = (1 - hw) / 2, (1 - hh) / 3
    for c, name in enumerate(COLS):
        ax.text(hw + cw * (c + 0.5), 1 - hh / 2, name, ha="center", va="center",
                fontsize=L["small_fs"], color=INK, fontweight="bold", linespacing=1.2)
    for r, name in enumerate(ROWS):
        y0 = 1 - hh - ch * (r + 1)
        ax.text(hw - 0.015, y0 + ch / 2, name, ha="right", va="center",
                fontsize=L["small_fs"], color=INK, fontweight="bold", linespacing=1.2)
        for c in range(2):
            x0 = hw + cw * c
            items = CELLS[(r, c)]
            gap = items is None
            ax.add_patch(Rectangle((x0, y0), cw, ch, fc=HL_FC if gap else "white",
                                   ec=HL if gap else GRID, lw=1.6 if gap else 0.8, zorder=1 if not gap else 2))
            if gap:
                ax.text(x0 + cw / 2, y0 + ch / 2,
                        "This work\nheadroom measured separately from\nrelative metrics, across students\ndiffering only in loss",
                        ha="center", va="center", fontsize=L["small_fs"], color=HL, linespacing=1.25, zorder=3)
            elif items:
                ax.text(x0 + cw / 2, y0 + ch / 2, "\n".join(items), ha="center", va="center",
                        fontsize=L["small_fs"], color=INK2, linespacing=1.25, zorder=3)
            else:
                ax.text(x0 + cw / 2, y0 + ch / 2, "—", ha="center", va="center",
                        fontsize=L["small_fs"], color=GRID, zorder=3)

    out = args.out + ("_single" if args.layout == "single" else "")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"→ {out}.png/.pdf")


if __name__ == "__main__":
    main()
