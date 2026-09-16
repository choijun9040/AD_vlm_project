"""
그림 2 — 입력 해상도에 따른 활성 크기와 fp16 붕괴율
======================================================
`_orchestrate_resolution_sweep.sh`가 남긴 `eval_results/resolution_sweep/res_*.json`을
읽어 두 개의 패널을 그린다.

  (a) 마지막 vision block의 max|activation| 중앙값 vs 입력 해상도 + fp16 표현 한계선
  (b) 이미지 단위 fp16 붕괴(NaN) 발생률 vs 입력 해상도

(a)가 원인, (b)가 결과다. 여유가 없는 모델만 한계선을 넘고 그 순간 붕괴율이 치솟는다.
x축은 max_pixels 대신 실제 vision token 수로 둔다 — 활성 크기를 좌우하는 것이
픽셀 수가 아니라 self-attention이 처리하는 토큰 수이기 때문이다.

실행:
    python scripts/plot_resolution_collapse.py --layout double
    python scripts/plot_resolution_collapse.py --layout single
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FP16_MAX = 65504.0
PATCH_PX = 28 * 28        # merge 후 토큰 하나가 덮는 픽셀 수
EVAL_TOKENS = 200704 // PATCH_PX      # 학습·평가에 쓴 해상도(256 토큰)

SERIES = [
    ("student_baseline_v2", "Task CE only",       "Task only",          "#b03a1f", "-",  2.4, "o"),
    ("student_temporal",    "Task + $L_{atc}$",   "Task + $L_{atc}$",   "#8a97a8", "-",  1.3, "^"),
    ("base_3b",             "Pre-trained (no FT)", "Pre-trained",       "#6b7674", "--", 1.3, "s"),
    ("student_kd_only_v4",  "Task + output KD",   "Task + out.KD",      "#a99b6f", "-",  1.3, "v"),
    ("student_spatial",     "Task + $L_{align}$", "Task + $L_{align}$", "#1f8a70", "--", 1.8, "D"),
    ("student_full",        "Task + $L_{align}$ + $L_{atc}$",
                            "Task + $L_{align}$ + $L_{atc}$",           "#0d6b59", "-",  2.4, "o"),
]

LAYOUT = {
    "double": dict(figsize=(7.4, 3.4), base_fs=10, tick_fs=9, legend_fs=8.5,
                   lw_scale=1.0, ms=4.5, legend_below=False),
    "single": dict(figsize=(3.4, 4.6), base_fs=7.5, tick_fs=6.5, legend_fs=6.2,
                   lw_scale=0.62, ms=3.0, legend_below=True),
}


def load(sweep_dir):
    """{max_pixels: 프로파일 dict} 로 읽어들인다."""
    out = {}
    for p in sorted(Path(sweep_dir).glob("res_*.json")):
        mp = int(p.stem.split("_")[1])
        out[mp] = json.loads(p.read_text())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", default="eval_results/resolution_sweep")
    ap.add_argument("--out", default="figures/fig2_resolution_collapse")
    ap.add_argument("--layout", default="double", choices=list(LAYOUT))
    args = ap.parse_args()

    L = LAYOUT[args.layout]
    if args.layout != "double" and args.out == ap.get_default("out"):
        args.out = f"{args.out}_{args.layout}"

    data = load(args.sweep_dir)
    if not data:
        raise SystemExit(f"{args.sweep_dir}에 res_*.json이 없다 — 스윕이 아직 안 끝났다")
    mps = sorted(data)
    tokens = [mp // PATCH_PX for mp in mps]
    n_images = data[mps[0]].get("_meta", {}).get("n_images", "?")

    plt.rcParams.update({"font.size": L["base_fs"], "axes.labelsize": L["base_fs"],
                         "xtick.labelsize": L["tick_fs"], "ytick.labelsize": L["tick_fs"]})
    nrows, ncols = (2, 1) if args.layout == "single" else (1, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=L["figsize"])
    ax_a, ax_b = axes.ravel()

    def points(key, pick):
        """해상도별로 (토큰 수, 값)을 모은다. 아직 측정되지 않은 해상도는 건너뛴다."""
        xs, ys = [], []
        for mp in mps:
            entry = data[mp].get(key)
            if entry is None:
                continue
            xs.append(mp // PATCH_PX)
            ys.append(pick(entry))
        return xs, ys

    # --- (a) 마지막 블록 활성 크기 ---
    ax_a.axhline(FP16_MAX, color="#b03a1f", lw=1.3 * L["lw_scale"], ls=(0, (5, 3)), zorder=1)
    # 라벨을 **왼쪽 위**에 둔다 (2026-09-16). 오른쪽 아래(기존)는 Task CE only 곡선이
    # 한계선을 가로지르는 바로 그 지점이라 글자가 곡선에 관통당했다 — 그림에서 가장
    # 중요한 주석이 읽히지 않았다. 왼쪽 상단은 모든 곡선이 한계선 아래라 비어 있다.
    ax_a.text(tokens[0], FP16_MAX * 1.04, "fp16 limit (65,504)",
              fontsize=L["legend_fs"] + 0.5, color="#b03a1f", va="bottom", ha="left")

    for key, label_d, label_s, color, ls, lw, mk in SERIES:
        xs, ys = points(key, lambda e: e["passes"]["bfloat16"]["by_layer"][str(e["depth"] - 1)]["p50"])
        if not xs:
            continue
        ax_a.plot(xs, ys, color=color, ls=ls, lw=lw * L["lw_scale"], marker=mk,
                  ms=L["ms"], label=(label_s if args.layout == "single" else label_d), zorder=3)

    ax_a.set_yscale("log")
    ax_a.set_ylabel(r"max$|$activation$|$, final block")
    ax_a.set_title("(a) activation magnitude", fontsize=L["base_fs"], pad=6)

    # --- (b) 붕괴율 ---
    for key, label_d, label_s, color, ls, lw, mk in SERIES:
        xs, ys = points(key, lambda e: e["passes"]["float16"]["nan_rate"] * 100)
        if not xs:
            continue
        ax_b.plot(xs, ys, color=color, ls=ls, lw=lw * L["lw_scale"], marker=mk,
                  ms=L["ms"], zorder=3)

    ax_b.set_ylabel("fp16 collapse rate (% of images)")
    ax_b.set_ylim(-4, 104)
    ax_b.set_title("(b) generation collapse", fontsize=L["base_fs"], pad=6)

    for ax in (ax_a, ax_b):
        ax.set_xscale("log", base=2)
        ax.set_xticks(tokens)
        ax.set_xticklabels([str(t) for t in tokens])
        ax.set_xlabel("Vision tokens per image")
        ax.axvline(EVAL_TOKENS, color="#c9ccc8", lw=1.0, zorder=0)
        ax.grid(True, which="major", axis="y", color="#e4e8e4", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # 문구를 줄이고 위치를 낮춘다 (2026-09-16). 기존 "training / eval resolution"은
    # 세로로 서서 범례까지 올라가 위쪽이 잘렸다.
    ax_b.text(EVAL_TOKENS * 0.94, 1, "training / eval",
              rotation=90, va="bottom", ha="right",
              fontsize=L["legend_fs"] - 0.5, color="#8a918c")

    if L["legend_below"]:
        ax_b.legend(*ax_a.get_legend_handles_labels(), loc="upper center",
                    bbox_to_anchor=(0.5, -0.32), fontsize=L["legend_fs"], frameon=False,
                    ncol=2, handlelength=1.9, labelspacing=0.28, columnspacing=1.0,
                    borderaxespad=0.0)
    else:
        ax_b.legend(*ax_a.get_legend_handles_labels(), loc="upper left",
                    fontsize=L["legend_fs"], frameon=False,
                    handlelength=2.2, labelspacing=0.32, borderaxespad=0.6)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = Path(f"{args.out}.{ext}")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"저장: {out}")

    print(f"이미지 {n_images}장 기준, 해상도 {tokens} 토큰")
    for key, *_ in SERIES:
        xs, rates = points(key, lambda e: e["passes"]["float16"]["nan_rate"] * 100)
        if xs:
            print(f"  {key:22s} 붕괴율 " + " ".join(f"{r:6.1f}%" for r in rates)
                  + ("" if len(xs) == len(mps) else f"   (측정된 해상도 {len(xs)}/{len(mps)})"))


if __name__ == "__main__":
    main()
