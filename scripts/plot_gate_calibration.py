"""
판정 기준 보정 그림 (학위 논문 7장 7.3·7.4절) — 최악 여유 대 fp16 붕괴율.

`fig_detection_curve*.png`(2026-09-15)를 대체한다. 옛 그림은 게이트를 최악 통계량으로
통일하기(2026-09-17) 전이라 p50과 최악을 나란히 그리고 "current threshold 2.0"을 판정
기준처럼 표시했다. 지금 도구의 1차 판정은 임계값 없는 **초과 비율**이고 2.0은 `fix`의
**교정 목표치**일 뿐이므로, 그 표시가 틀린 정보를 준다.

이 그림이 보이는 것은 둘이다.
  (1) 전이는 형식 상한 그 자체(최악 여유 1.00~1.05)에서 일어난다 — 합성 연속체
  (2) 학습으로 만들어진 체크포인트(λ 스윕 넷)가 같은 곡선 위에 놓인다 — 자연 발생 점

데이터(모두 100장, 이미지를 줄이지 않는 원본 해상도 1600×900, 크기는 bf16·붕괴는 fp16):
  합성  eval_results/detection_curve.json, detection_curve_fine.json  (겹치는 배수는 한 번만)
  자연  eval_results/guard_sweep_align{0.0,0.5,1.0,2.0}.json           (마지막 블록 최악 여유, nan_rate)

색: dataviz 참조 팔레트 1·2번 슬롯(파랑·주황). validate_palette.js로 전 항목 통과 확인.
기준선은 시리즈 색이 아니라 중립 잉크로 그린다.

실행:
    python scripts/plot_gate_calibration.py                 # 2단 폭
    python scripts/plot_gate_calibration.py --layout single # 1단 폭
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

SYNTH = "#2a78d6"      # 참조 팔레트 slot 1
NATURAL = "#eb6834"    # 참조 팔레트 slot 2
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e4e1"
MARGINAL = 1.25        # headroom_guard.MARGINAL_HEADROOM (설계 판단)
FIX_TARGET = 2.0       # headroom_guard --target_headroom 기본값 (교정 목표)

LAYOUT = {
    "double": dict(figsize=(6.4, 3.3), base_fs=9.5, tick_fs=8.5, note_fs=8, lw=2.0, ms=7,
                   xlabel="Worst-image fp16 headroom  (65,504 / max|activation|, final block)",
                   legend=dict(loc="center right")),
    "single": dict(figsize=(3.4, 3.1), base_fs=7.5, tick_fs=6.5, note_fs=6.0, lw=1.6, ms=5.5,
                   xlabel="Worst-image fp16 headroom (final block)",
                   legend=dict(loc="center right", bbox_to_anchor=(1.0, 0.42))),
}


def synthetic_points():
    seen, pts = set(), []
    for f in ("eval_results/detection_curve.json", "eval_results/detection_curve_fine.json"):
        d = json.loads(Path(f).read_text())
        for r in d["rows"]:
            key = round(r["factor"], 3)
            if key in seen:
                continue
            seen.add(key)
            pts.append((r["headroom_worst"], r["collapse_rate"] * 100))
    return sorted(pts)


def natural_points():
    pts = []
    for lam in ("0.0", "0.5", "1.0", "2.0"):
        g = json.loads(Path(f"eval_results/guard_sweep_align{lam}.json").read_text())
        last = [b for b in g["before"]["per_block"] if b["block"] != "tower_out"][-1]
        pts.append((lam, last["headroom_worst"], g["before"]["nan_rate"] * 100))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="double", choices=list(LAYOUT))
    ap.add_argument("--out", default="figures/fig_gate_calibration")
    args = ap.parse_args()
    L = LAYOUT[args.layout]
    out = args.out if args.layout == "double" else f"{args.out}_{args.layout}"

    syn, nat = synthetic_points(), natural_points()
    plt.rcParams.update({"font.size": L["base_fs"], "axes.labelsize": L["base_fs"],
                         "xtick.labelsize": L["tick_fs"], "ytick.labelsize": L["tick_fs"],
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=L["figsize"])

    # 경계 구간(1.00~1.25) — 설계 판단이므로 옅게, 이름을 붙여 둔다
    ax.axvspan(1.0, MARGINAL, color="#f1efe9", zorder=0, lw=0)
    ax.axvline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axvline(FIX_TARGET, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)

    xs, ys = zip(*syn)
    ax.plot(xs, ys, color=SYNTH, lw=L["lw"], marker="o", ms=L["ms"] - 1.5,
            markeredgecolor="white", markeredgewidth=0.8, zorder=3,
            label=f"Synthetic continuum (MLP scaled, {len(syn)} points)")
    ax.scatter([p[1] for p in nat], [p[2] for p in nat], s=(L["ms"] + 2.5) ** 2, marker="D",
               color=NATURAL, edgecolor="white", linewidth=1.0, zorder=4,
               label="Trained checkpoints ($\\lambda_{align}$ sweep)")
    # 라벨 위치를 점마다 정한다 — 0.5와 1.0은 가까워 같은 높이면 겹친다
    offsets = {"0.0": (-7, 0, "right", "center"), "0.5": (0, 9, "center", "bottom"),
               "1.0": (0, 21, "center", "bottom"), "2.0": (0, 9, "center", "bottom")}
    for lam, h, c in nat:
        dx, dy, ha, va = offsets[lam]
        ax.annotate(f"$\\lambda$={lam}", (h, c), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=L["note_fs"], color=INK2)

    ax.set_xscale("log")
    ticks = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.72, 4.6)
    ax.set_ylim(-4, 100)
    ax.set_xlabel(L["xlabel"])
    ax.set_ylabel("fp16 collapse rate (% of images)")
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.text(1.0, 97, " fp16 limit", fontsize=L["note_fs"], color=INK, va="top", ha="left")
    ax.text((1.0 * MARGINAL) ** 0.5, 84, "marginal\n(design\nchoice)", fontsize=L["note_fs"] - 0.5,
            color=INK2, ha="center", va="top")
    ax.text(FIX_TARGET, 97, " fix target 2.0", fontsize=L["note_fs"], color=INK2, va="top", ha="left")

    ax.legend(fontsize=L["note_fs"], frameon=False, handletextpad=0.5, **L["legend"])
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"합성 {len(syn)}점 · 자연 {len(nat)}점 → {out}.png/.pdf")
    first_safe = min(h for h, c in syn if c == 0)
    print(f"합성 최초 무붕괴 최악 여유 {first_safe:.3f} · 자연: "
          + ", ".join(f"λ{l} {h:.2f}/{c:.0f}%" for l, h, c in nat))


if __name__ == "__main__":
    main()
