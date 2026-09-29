"""
지표 역전 산점도 (학위 논문 4.4절, 그림 4-4) — fp16 여유 대 INT8 양자화 SNR.

4.4절 표(Table 1)의 여섯 모델을 한 그림에 놓는다. 가로축은 마지막 블록의 fp16 여유(이미지별 최대의
중앙값, bf16에서 측정), 세로축은 같은 블록 텐서의 INT8 양자화 SNR이다. 여유가 큰(안전한) 모델일수록
SNR이 낮게(위험하게) 평가된다는 것이 기여의 실체이므로, 그것을 한눈에 보이게 하는 그림이다.

  (a) 텐서 단위 SNR — 여유(p50)와 완전히 역순이다(6모델 Spearman ρ = −1.000)
  (b) 채널 단위 SNR — 정렬만에서 단조성이 깨진다. 두 끝점의 역전은 남는다

**정직하게 둘을 나란히 둔다.** (a)만 보이면 입도를 고른 것으로 읽힌다(4.4절).
최악값 통계량으로 여유를 재면 인접 한 쌍(시간만·출력 증류)이 뒤집혀 5변형 ρ = −0.900이다 —
캡션에 적는다(그림은 표와 같은 p50).

표현: 여섯 개체를 **색으로 구별하지 않는다.** 산점도는 모든 쌍이 인접하므로(dataviz: 범주색은
all-pairs에서 3개까지) 개체는 **점마다 직접 붙인 이름**으로 식별하고, 채움으로 붕괴 여부를 나타낸다
(채움 = 원본 해상도 fp16에서 붕괴 > 0, 속 빈 점 = 0%). 점과 글자는 중립 잉크.

데이터(원본 해상도 max_pixels 1,440,000):
  여유·붕괴율  eval_results/resolution_sweep/res_1440000.json   (250장)
  INT8 SNR    eval_results/int8_quant_error_native.json          (200장)

실행:
    python scripts/plot_metric_inversion.py                 # 2단 폭
    python scripts/plot_metric_inversion.py --layout single # 1단 폭
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spearman(x, y):
    """순위 상관 (동순위 없음 가정 — 여섯 값 모두 서로 다르다). scipy 의존을 피한다."""
    assert len(set(x)) == len(x) and len(set(y)) == len(y), "동순위가 있으면 평균 순위가 필요하다"
    rx = {v: i for i, v in enumerate(sorted(x))}
    ry = {v: i for i, v in enumerate(sorted(y))}
    n = len(x)
    d2 = sum((rx[a] - ry[b]) ** 2 for a, b in zip(x, y))
    return 1 - 6 * d2 / (n * (n * n - 1))

FP16_MAX = 65504.0
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e1"

# (키, 이름, 라벨 오프셋(점) — 가까운 점끼리 겹치지 않게 개별로 정한다)
MODELS = [
    ("student_baseline_v2", "Task only"),
    ("base_3b",             "Pre-trained"),
    ("student_temporal",    "Task + $L_{atc}$"),
    ("student_kd_only_v4",  "Task + out.KD"),
    ("student_spatial",     "Task + $L_{align}$"),
    ("student_full",        "Full"),
]
# 라벨 자리 (2026-09-29 렌더 후 조정). 1.0 근처 네 점은 서로 붙어 있어 점 옆에 달면 글자가 겹친다 —
# 가운데 빈 자리(x=LABEL_X)에 세로로 벌려 놓고 가는 인출선으로 잇는다. 오른쪽 두 점은 점 옆에 단다.
LABEL_X = 1.32
CLUSTER_Y = {  # panel -> key -> 라벨 y (데이터 좌표)
    "a": {"student_baseline_v2": 32.25, "base_3b": 31.55, "student_temporal": 30.85,
          "student_kd_only_v4": 30.15},
    "b": {"student_baseline_v2": 40.56, "base_3b": 39.60, "student_temporal": 38.05,
          "student_kd_only_v4": 37.30},
}
SIDE = {  # panel -> key -> (dx, dy, ha, va) 오프셋(점)
    "a": {"student_spatial": (-9, 0, "right", "center"), "student_full": (-9, 0, "right", "center")},
    "b": {"student_spatial": (0, -11, "center", "top"), "student_full": (-9, 0, "right", "center")},
}

LAYOUT = {
    "double": dict(figsize=(7.2, 3.3), base_fs=9, tick_fs=8, note_fs=7.5, ms=7.5, rows=1),
    "single": dict(figsize=(3.4, 5.4), base_fs=7.5, tick_fs=6.5, note_fs=6.2, ms=6, rows=2),
}


def load():
    res = json.loads(Path("eval_results/resolution_sweep/res_1440000.json").read_text())
    q = json.loads(Path("eval_results/int8_quant_error_native.json").read_text())
    rows = []
    for key, name in MODELS:
        p50 = res[key]["passes"]["bfloat16"]["by_layer"]["31"]["p50"]
        s = q[key]["by_layer"]["31"]
        rows.append(dict(key=key, name=name, headroom=FP16_MAX / p50,
                         collapse=res[key]["passes"]["float16"]["nan_rate"] * 100,
                         snr_t=s["snr_tensor"]["p50"], snr_c=s["snr_channel"]["p50"]))
    return rows, res["_meta"]["n_images"], q["_meta"]["n_images"]


def panel(ax, rows, field, title, tag, L):
    ax.axvline(1.0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    for r in rows:
        filled = r["collapse"] > 0
        ax.plot(r["headroom"], r[field], marker="o", ms=L["ms"], ls="none", zorder=3,
                markerfacecolor=INK if filled else "white", markeredgecolor=INK,
                markeredgewidth=1.3)
        lab = f"{r['name']}  {r['collapse']:.1f}%" if filled else f"{r['name']}  0%"
        if r["key"] in CLUSTER_Y[tag]:
            ax.annotate(lab, (r["headroom"], r[field]), xytext=(LABEL_X, CLUSTER_Y[tag][r["key"]]),
                        textcoords="data", ha="left", va="center", fontsize=L["note_fs"], color=INK2,
                        arrowprops=dict(arrowstyle="-", color="#9a9994", lw=0.6,
                                        shrinkA=2, shrinkB=L["ms"] * 0.7))
        else:
            dx, dy, ha, va = SIDE[tag][r["key"]]
            ax.annotate(lab, (r["headroom"], r[field]), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=L["note_fs"], color=INK2)
    rho = spearman([r["headroom"] for r in rows], [r[field] for r in rows])
    ax.set_title(f"{title}   (Spearman ρ = {rho:+.3f}, n = {len(rows)})",
                 fontsize=L["base_fs"], pad=6, color=INK)
    ax.set_xlim(0.7, 3.6)
    ax.set_xlabel("fp16 headroom, final block (median over images)")
    ax.set_ylabel("INT8 quantization SNR (dB)")
    ax.grid(True, axis="y", color=GRID, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(0.985, ax.get_ylim()[0], "fp16 limit ", fontsize=L["note_fs"], color=INK,
            va="bottom", ha="right", rotation=90)
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="double", choices=list(LAYOUT))
    ap.add_argument("--out", default="figures/fig4_metric_inversion")
    args = ap.parse_args()
    L = LAYOUT[args.layout]
    out = args.out if args.layout == "double" else f"{args.out}_{args.layout}"

    rows, n_res, n_q = load()
    plt.rcParams.update({"font.size": L["base_fs"], "axes.labelsize": L["base_fs"],
                         "xtick.labelsize": L["tick_fs"], "ytick.labelsize": L["tick_fs"],
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    nr, nc = (2, 1) if L["rows"] == 2 else (1, 2)
    fig, (ax_a, ax_b) = plt.subplots(nr, nc, figsize=L["figsize"])
    ra = panel(ax_a, rows, "snr_t", "(a) per-tensor", "a", L)
    rb = panel(ax_b, rows, "snr_c", "(b) per-channel", "b", L)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"여유·붕괴 {n_res}장, SNR {n_q}장 → {out}.png/.pdf  ρ(텐서) {ra:+.3f} · ρ(채널) {rb:+.3f}")
    for r in sorted(rows, key=lambda r: r["headroom"]):
        print(f"  {r['key']:22} 여유 {r['headroom']:.2f}  붕괴 {r['collapse']:5.1f}%  "
              f"SNR 텐서 {r['snr_t']:.2f} · 채널 {r['snr_c']:.2f}")


if __name__ == "__main__":
    main()
