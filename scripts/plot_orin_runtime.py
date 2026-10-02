"""
실행 경로별 붕괴율과 엣지 보드 지연 (학위 논문 6.9절, 그림 6-1).

  (a) 기준선 학생의 fp16 붕괴율을 실제 비전 토큰 수(480 · 966 · 1,824)에 대해 실행 경로별로 겹친다.
      초과 비율(bf16 추정) · PyTorch fp16 · TensorRT 강타입(A100) · TensorRT 관행 `--fp16`(A100) · Orin 강타입.
      강타입은 PyTorch를 따라가고 관행 빌드만 해상도에 따라 양쪽으로 어긋난다는 것, Orin 점이 A100 강타입에
      겹친다는 것이 그림의 요지다. Full과 교정본은 모든 경로·해상도에서 0%라 선으로 그리지 않고 주석으로 적는다.
  (b) Orin Nano 지연(p50, 분할 엔진 두 개의 합). 원본 해상도는 엔진을 빌드하지 못했다 — 그렇게 표시한다.

표현: 경로 넷을 색으로 구별하되(참조 팔레트 slot 1·2·3, `validate_palette.js` 통과 — 청록은 대비 WARN이라
모든 계열에 이름을 직접 붙이고 표지 모양을 달리한다), Orin은 A100 강타입과 같은 주황에 속 빈 큰 표지로 둔다
(같은 엔진 조건, 다른 기기). 초과 비율은 측정이 아니라 추정이므로 중립 잉크 점선이다.

데이터:
  초과 비율·PyTorch   eval_results/resolution_sweep/res_{1440000,802816,401408}.json   (250장)
  TRT 강타입          eval_results/trt_{nativeh,802816h,401408h}_typed_baseline_v2.json (250장, TRT 10.13)
  TRT 관행 (fp32 그래프) eval_results/trt_native250_baseline_v2_dense.json, trt_802816_baseline_v2.json,
                      trt_401408_baseline_v2_single.json                                  (250장, TRT 10.13)
  Orin               eval_results/orin/typed_results/orin_*_typed.json                   (250장, TRT 10.3, 15W)

실행:
    python scripts/plot_orin_runtime.py                 # 2단 폭
    python scripts/plot_orin_runtime.py --layout single # 1단 폭
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYT = "#2a78d6"      # 참조 팔레트 slot 1
TYPED = "#eb6834"    # slot 2 — A100 강타입과 Orin 강타입이 공유한다
WEAK = "#1baf7a"     # slot 3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e1"

RES = [(401408, "401408"), (802816, "802816"), (1440000, "native")]
LAYOUT = {
    "double": dict(figsize=(7.2, 3.3), base_fs=9, tick_fs=8, note_fs=7.5, lw=2.0, ms=6.5, rows=1),
    "single": dict(figsize=(3.4, 5.6), base_fs=7.5, tick_fs=6.5, note_fs=6.2, lw=1.6, ms=5, rows=2),
}


def j(p):
    return json.loads(Path(p).read_text())


def load():
    tokens, exceed, pyt, typed, weak = [], [], [], [], []
    weak_files = {"401408": "trt_401408_baseline_v2_single", "802816": "trt_802816_baseline_v2",
                  "native": "trt_native250_baseline_v2_dense"}
    for mp, tag in RES:
        r = j(f"eval_results/resolution_sweep/res_{mp}.json")
        b = r["student_baseline_v2"]
        exceed.append(b["passes"]["bfloat16"]["by_layer"]["31"]["over_fp16_rate"] * 100)
        pyt.append(b["passes"]["float16"]["nan_rate"] * 100)
        t = j(f"eval_results/trt_{tag}h_typed_baseline_v2.json")
        assert t["n_images"] == 250
        typed.append(t["modes"]["typed"]["nan_rate"] * 100)
        w = j(f"eval_results/{weak_files[tag]}.json")
        assert w["n_images"] == 250
        weak.append(w["modes"]["fp16"]["nan_rate"] * 100)
    # 실제 비전 토큰 수(merge 후) — 4장 그림 4-2와 같은 값. 1600×900 이미지의 smart_resize 결과다
    tokens = [480, 966, 1824]
    orin = {}
    for tag, tok in (("401408", 480), ("802816", 966)):
        f = "eval_results/orin/typed_results/orin_{}_typed.json"
        d = {m: j(f.format(f"{m}_{tag}h")) for m in ("baseline_v2", "full")}
        d["fixa"] = j(f.format(f"baseline_v2_{tag}h_fixa"))
        b = d["baseline_v2"]
        assert b["n"] == 250
        orin[tok] = dict(collapse=b["nan_rate"] * 100, n=b["nan_count"], lat=b["latency_ms"]["p50"],
                         others=[v["nan_rate"] for k, v in d.items() if k != "baseline_v2"])
    return tokens, exceed, pyt, typed, weak, orin


def label(ax, x, y, text, color, L, dx=6, dy=0, ha="left", va="center"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=L["note_fs"], color=INK2)
    # 텍스트는 잉크색, 계열 식별은 바로 옆 선·표지가 맡는다(색만으로 구별하지 않는다)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="double", choices=list(LAYOUT))
    ap.add_argument("--out", default="figures/fig6_runtime_collapse")
    args = ap.parse_args()
    L = LAYOUT[args.layout]
    out = args.out if args.layout == "double" else f"{args.out}_{args.layout}"

    tokens, exceed, pyt, typed, weak, orin = load()
    plt.rcParams.update({"font.size": L["base_fs"], "axes.labelsize": L["base_fs"],
                         "xtick.labelsize": L["tick_fs"], "ytick.labelsize": L["tick_fs"],
                         "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    nr, nc = (2, 1) if L["rows"] == 2 else (1, 2)
    fig, (a, b) = plt.subplots(nr, nc, figsize=L["figsize"],
                               gridspec_kw={"width_ratios": [1.6, 1]} if nc == 2 else None)

    # (a) 붕괴율
    a.plot(tokens, exceed, color=INK2, lw=1.0, ls=(0, (3, 2)), marker="x", ms=L["ms"] - 1, zorder=2)
    a.plot(tokens, pyt, color=PYT, lw=L["lw"], marker="o", ms=L["ms"], zorder=3)
    a.plot(tokens, typed, color=TYPED, lw=L["lw"], marker="s", ms=L["ms"] - 0.5, zorder=4)
    a.plot(tokens, weak, color=WEAK, lw=L["lw"], marker="^", ms=L["ms"], zorder=3)
    ox = sorted(orin)
    a.plot(ox, [orin[x]["collapse"] for x in ox], ls="none", marker="D", ms=L["ms"] + 5,
           markerfacecolor="none", markeredgecolor=TYPED, markeredgewidth=1.6, zorder=5)
    # 직접 라벨 — 원본 해상도 끝점 오른쪽에. 위 둘(초과·PyTorch·강타입)이 92~96%에 몰려 세로로 벌린다
    xe = tokens[-1]
    sp = L["note_fs"] * 1.9                       # 원본 해상도 끝점 라벨 셋의 세로 간격(점)
    label(a, xe, exceed[-1], "bf16 exceedance (estimate)", INK2, L, dx=10, dy=sp)
    label(a, xe, pyt[-1], "PyTorch fp16", PYT, L, dx=10, dy=0)
    label(a, xe, typed[-1], "TensorRT strongly typed", TYPED, L, dx=10, dy=-sp)
    label(a, tokens[1], weak[1], "TensorRT --fp16 (weak)", WEAK, L, dx=10, dy=0)
    label(a, ox[-1], orin[ox[-1]]["collapse"], "Orin Nano\n(strongly typed)", TYPED, L, dx=-14, dy=10,
          ha="right", va="bottom")
    a.text(0.98, 0.45 if nc == 2 else 0.30, "Full and corrected engine:\n0% on every path",
           transform=a.transAxes, fontsize=L["note_fs"], color=INK2, va="center", ha="right")
    a.set_xlim(300 if nc == 2 else -100, 2900 if nc == 2 else 3000)   # 1단은 왼쪽 라벨 자리를 더 둔다
    a.set_ylim(-3, 112)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_xticks(tokens)
    a.set_xlabel("Vision tokens per image")
    a.set_ylabel("fp16 collapse rate (%), baseline student")
    a.set_title("(a) collapse rate by execution path", fontsize=L["base_fs"], color=INK, pad=6)

    # (b) Orin 지연
    lat = [orin[x]["lat"] for x in ox]
    b.plot(ox, lat, color=TYPED, lw=L["lw"], marker="D", ms=L["ms"] + 1,
           markerfacecolor="white", markeredgecolor=TYPED, markeredgewidth=1.6, zorder=3)
    for x, y in zip(ox, lat):
        b.annotate(f"{y:,.0f} ms", (x, y), xytext=(9, -2), textcoords="offset points",
                   ha="left", va="center", fontsize=L["note_fs"], color=INK2)
    b.axvline(tokens[-1], color=INK2, lw=0.8, ls=(0, (2, 2)))
    b.text(tokens[-1], 0.5, "engine build failed\n(out of memory)", transform=b.get_xaxis_transform(),
           rotation=90, ha="right", va="center", fontsize=L["note_fs"], color=INK2)
    b.set_xlim(300, 2000)
    b.set_ylim(0, 1400)
    b.set_xticks(tokens)
    b.set_xlabel("Vision tokens per image")
    b.set_ylabel("Orin Nano latency p50 (ms)")
    b.set_title("(b) edge-board latency", fontsize=L["base_fs"], color=INK, pad=6)

    for ax in (a, b):
        ax.grid(True, axis="y", color=GRID, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"→ {out}.png/.pdf")
    print("  토큰      초과    PyTorch  강타입  관행    Orin")
    for i, t in enumerate(tokens):
        o = orin.get(t)
        ob = f"{o['collapse']:.1f} ({o['n']}장, {o['lat']:.0f} ms)" if o else "빌드 불가"
        print(f"  {t:5d}  {exceed[i]:6.1f}  {pyt[i]:6.1f}  {typed[i]:6.1f}  {weak[i]:6.1f}  {ob}")
    print("  Orin Full·교정본 붕괴율:", {t: orin[t]["others"] for t in ox})


if __name__ == "__main__":
    main()
