"""원고 전면 수정(#11) 그림 — 결과 파일에서 다시 만든다. 2026-11-09.

  fig_resid_schematic   그림 3-1  잔차 재조정(RR) 개념도 (데이터 없음)
  fig_calib_c           그림 5-2  최댓값 M과 c (7개 조건)
  fig_fidelity          그림 5-3  원래 fp32 출력 대비 상대 오차 — bf16·기존 처방·RR
  fig_deploy            그림 5-4  배포 경로별 상실(무교정 대 RR)과 A100 TensorRT 지연

그림 속 글자는 영문, 캡션은 국문(thesis_figures.md).

    python scripts/plot_thesis_v2.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "eval_results/cmp"
OUT = ROOT / "figures"
COL = {"fp16": "#c0392b", "bf16": "#8e959b", "prior": "#e08e2b", "R": "#1f5fa8", "fp32": "#2f2f2f"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
J = lambda f: json.load(open(C / f))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("저장", OUT / f"{name}.png")


def schematic():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
    y = 22
    ax.annotate("", xy=(97, y), xytext=(3, y), arrowprops=dict(arrowstyle="-|>", lw=2.2, color=COL["R"]))
    ax.text(50, y + 9.5, "residual stream  h → c·h   (|c·h| ≤ 65,504 / 2)", ha="center", color=COL["R"], fontsize=9.5, fontweight="bold")
    box = lambda x, w, t, col: (ax.add_patch(FancyBboxPatch((x, 4), w, 8, boxstyle="round,pad=0.4", fc="white", ec=col, lw=1.3)),
                                ax.text(x + w / 2, 8, t, ha="center", va="center", fontsize=8.2))
    box(1, 17, "entry\n(embedding) ×c", COL["R"])
    ax.add_patch(FancyArrowPatch((9.5, 12.5), (4, y - 0.6), arrowstyle="-|>", mutation_scale=9, color=COL["R"], lw=1.2))   # 입력부가 흐름의 시작
    for x0 in (24, 54):
        box(x0, 10, "norm\neps ×c²", COL["bf16"])
        box(x0 + 13, 10, "block f(·)", COL["bf16"])
        ax.add_patch(FancyArrowPatch((x0 - 1, y), (x0 + 5, 12.5), arrowstyle="-|>", mutation_scale=9, color="#555", lw=1))
        ax.add_patch(FancyArrowPatch((x0 + 10.5, 8), (x0 + 13, 8), arrowstyle="-|>", mutation_scale=9, color="#555", lw=1))
        ax.add_patch(FancyArrowPatch((x0 + 18, 12.5), (x0 + 21, y - 0.6), arrowstyle="-|>", mutation_scale=9, color=COL["R"], lw=1.2))
        ax.text(x0 + 17.5, y - 4.6, "writer ×c", color=COL["R"], fontsize=7.8, ha="right")
        ax.text(x0 + 21.2, y + 1.2, "+", color=COL["R"], fontsize=11, ha="center")
    ax.text(88, 8, "exit ×1/c\n(only if read by a\nnon-norm layer)", ha="center", va="center", fontsize=7.8, color="#555")
    ax.text(50, 0.2, "norm(c·h) with eps·c²  =  norm(h)   ⇒   fp32 output unchanged; only the residual magnitude shrinks by c",
            ha="center", fontsize=8, color="#333")
    save(fig, "fig_resid_schematic")


def calib_c():
    """그림 5-2: 측정용 최댓값 M과 c (2026-11-10)."""
    rows = [("Qwen2.5-VL-3B", J("env4/dlm_gen_qwen.json.calib.json")["max_worst"], J("dlm_diag_qwen.json")["diagnosis_bf16"]["max_worst"],
             J("env4/dlm_gen_qwen.json.calib.json")["c"])]
    for m, mm in (("26B-A4B", "Gemma 4 26B-A4B"), ("31B", "Gemma 4 31B")):
        for t in (280, 1120):
            cal = J(f"calib100_gemma_{m}_{t}.json")
            rows.append((f"{mm}·{t}", cal["max_worst"], J(f"dlm_diag_gemma_{m}_{t}.json")["diagnosis_bf16"]["max_worst"], cal["c"]))
    sc = J("sdxl_calib.json"); rows.append(("SDXL VAE dec.", sc["max_worst"], J("env3/sdxl_gen_cmp.json")["diag_orig"]["max_worst"], sc["c"]))
    rows.append(("FLUX.1-schnell", J("flux_calib.json")["diagnosis_bf16"]["max_worst"], J("flux_cmp.json")["diagnosis_bf16"]["max_worst"], J("flux_rev35.json")["c"]))
    fig, ax = plt.subplots(figsize=(7.2, 2.7))                          # 2026-11-10 사용자 「평가 최댓값 생략, 그림은 안 1」: 측정용 M과 c만
    x = range(len(rows))
    ax.bar(list(x), [r[1] for r in rows], 0.55, color="#4a4f55", label="max $M$ (100 diagnosis inputs)")
    for i, r in enumerate(rows):
        ax.text(i, r[1] * 1.15, f"$c$ = 1/{round(1 / r[3])}", ha="center", va="bottom", fontsize=7)
    ax.axhline(65504, color=COL["fp16"], lw=1, ls="--", label="fp16 limit 65,504"); ax.axhline(32752, color="#333", lw=0.9, label="target 32,752")
    ax.set_yscale("log"); ax.set_ylim(2e4, 5e6)
    ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows], fontsize=7.4, rotation=12)
    ax.set_ylabel("max |activation| (bf16)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3, frameon=False, fontsize=7.6)
    save(fig, "fig_calib_c")


def fidelity():
    q, f, g = J("env4/dlm_fid_qwen.json"), J("flux_rev35.json"), J("env3/sdxl_gen_cmp.json")   # Qwen 개정 40·SDXL 개정 39(TF32 끔), FLUX 개정 35(같은 실행)
    b = J("env5/sdxl_gen_bf16.json")                                   # 개정 41: SDXL 원래 VAE bf16 (vast3, 같은 환경 판정 통과)
    groups = [("SDXL VAE\ndecoder\n(100)", [("bf16", b["orig_bf16"]["rel_err"]["median"], "bf16"), ("FP16-Fix", g["fix_fp16"]["rel_err"]["median"], "prior"), ("RR", g["resid_eps1_fp16"]["rel_err"]["median"], "R")]),
              ("FLUX.1-\nschnell\n(100)", [("bf16", f["bf16"]["rel_err_vs_fp32"]["median"], "bf16"),
                                                 ("Draw Things", f["b2_drawthings_engine"]["rel_err_vs_fp32"]["median"], "prior"),   # 2026-11-11 (b2) 하나만
                                                 ("RR", f["c_resid"]["rel_err_vs_fp32"]["median"], "R")]),
              ("Qwen2.5-\nVL-3B\n(799)", [("bf16", q["bf16"]["rel_err"]["median"], "bf16"), ("RR", q["fp16_resid"]["rel_err"]["median"], "R")])]
    for m, mm in (("26B-A4B", "Gemma 4 26B-A4B"), ("31B", "Gemma 4 31B")):
        for t in (280, 1120):
            d = J(f"dlm_fid_gemma_{m}_{t}.json")
            groups.append(("Gemma 4\n" + mm.split()[-1] + f"·{t}\n(799)", [("bf16", d["bf16"]["rel_err"]["median"], "bf16"), ("RR", d["fp16_resid"]["rel_err"]["median"], "R")]))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    pos, ticks, seen = 0.0, [], set()
    for name, bars in groups:
        xs = [pos + i * 0.8 for i in range(len(bars))]
        for xx, (lab, v, k) in zip(xs, bars):
            ax.bar(xx, v, 0.72, color=COL[k], hatch={"Draw Things": "///", "Draw Things (fp16 acc.)": "xxx"}.get(lab), edgecolor="white" if lab.startswith("Draw Things") else None,
                   label=None if lab in seen else lab)
            seen.add(lab)
            ax.text(xx, v * 1.12, f"{v:#.3g}", ha="center", fontsize=6.4)   # 이웃 막대 라벨과 겹치지 않게
        ticks.append((sum(xs) / len(xs), name)); pos = xs[-1] + 1.5
    ax.set_yscale("log"); ax.set_ylim(1e-3, 1.0)
    ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=7)
    ax.set_ylabel("rel. error vs. original fp32\n(median, log)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False, fontsize=8)
    save(fig, "fig_fidelity")


def deploy():
    orin = {k: json.load(open(ROOT / f"eval_results/orin_dlm/results/orin_official_dlm1596h_{k}.json")) for k in ("typed", "resid_typed")}
    loss = [("PyTorch\n(A100)", 100 * J("dlm_diag_qwen.json")["fp16_actual_loss"]["rate"], 0.0),
            ("TensorRT\n(A100)", 100 * J("dlm_lat_f16.json")["modes"]["typed"]["nan_rate"], 100 * J("dlm_lat_f16_resid.json")["modes"]["typed"]["nan_rate"]),
            ("TensorRT\n(Orin Nano)", 100 * orin["typed"]["nan_rate"], 100 * orin["resid_typed"]["nan_rate"])]
    lat = [("fp32", J("dlm_lat_f32.json")["modes"]["fp32"]["latency_ms"]["p50"], "fp32"), ("bf16", J("dlm_lat_bf16.json")["modes"]["bf16"]["latency_ms"]["p50"], "bf16"),
           ("fp16\n(uncorr.)", J("dlm_lat_f16.json")["modes"]["typed"]["latency_ms"]["p50"], "fp16"), ("fp16+RR", J("dlm_lat_f16_resid.json")["modes"]["typed"]["latency_ms"]["p50"], "R")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    w = 0.36
    for i, (n, u, r) in enumerate(loss):
        a1.bar(i - w / 2, u, w, color=COL["fp16"], label="fp16 (uncorrected)" if i == 0 else None)
        a1.bar(i + w / 2, r, w, color=COL["R"], label="fp16 + RR" if i == 0 else None)
        a1.text(i - w / 2, u + 2, f"{u:.1f}", ha="center", fontsize=7); a1.text(i + w / 2, r + 2, f"{r:.0f}", ha="center", fontsize=7)
    a1.set_xticks(range(3)); a1.set_xticklabels([l[0] for l in loss], fontsize=7.6); a1.set_ylabel("output loss (%)"); a1.set_ylim(0, 105)
    a1.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2); a1.set_title("(a) loss, 799 images", fontsize=8.5)
    for i, (n, v, k) in enumerate(lat):
        a2.bar(i, v, 0.6, color=COL[k]); a2.text(i, v + 3, f"{v:.1f}", ha="center", fontsize=7)
    a2.set_xticks(range(4)); a2.set_xticklabels([l[0] for l in lat], fontsize=7.6); a2.set_ylabel("latency p50 (ms)"); a2.set_ylim(0, 225)
    a2.set_title("(b) A100 TensorRT latency", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig_deploy")


KFONT = "Noto Sans CJK JP"                                   # 한국어 라벨 개념도(그림 2-1, 3-0)용


def teaser():
    """그림 1-1: (a) 세 형식의 비트 구성과 상한, (b) SDXL VAE 디코더 복호 예 — 원본 fp32 / fp16 무교정 / fp16 + RR."""
    from PIL import Image
    fig = plt.figure(figsize=(7.2, 2.5))
    ax = fig.add_axes([0.0, 0.0, 0.52, 1.0]); ax.set_xlim(0, 100); ax.set_ylim(5, 40); ax.axis("off")
    fmts = [("fp32", 8, 23, "max ≈ 3.4×10³⁸"), ("bf16", 8, 7, "max ≈ 3.4×10³⁸"), ("fp16", 5, 10, "max = 65,504")]
    unit = 62 / 32
    for k, (n, e, m, mx) in enumerate(fmts):
        y = 30 - k * 11
        ax.text(0, y + 2.2, n, fontsize=9, fontweight="bold", va="center")
        x = 11
        for w, col, lab in ((1, "#2f2f2f", "s"), (e, COL["bf16"], f"exp {e}"), (m, "#d9d9d9", f"mantissa {m}" if m > 8 else f"mant. {m}")):
            ax.add_patch(plt.Rectangle((x, y), w * unit, 4.4, fc=col, ec="white", lw=0.8))
            if w > 1:
                ax.text(x + w * unit / 2, y + 2.2, lab, ha="center", va="center", fontsize=6.8, color="white" if col != "#d9d9d9" else "#333")
            x += w * unit
        ax.text(x + 1.5, y + 2.2, mx, fontsize=7.4, va="center", color=COL["fp16"] if n == "fp16" else "#333")
    ax.text(0, 38.5, "(a) 16-bit formats and their limits", fontsize=8.5)
    G = Image.open(OUT / "fig_sdxl_gen_view.png").convert("RGB"); col = 3; s = 256
    tiles = [(G.crop((s * col, s * r, s * col + s, s * r + s)), lab) for r, lab in ((0, "original fp32"), (1, "fp16 (uncorrected)"), (3, "fp16 + RR"))]
    for j, (im, lab) in enumerate(tiles):
        a = fig.add_axes([0.54 + j * 0.153, 0.14, 0.145, 0.72]); a.imshow(im); a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_color(COL["fp16"] if j == 1 else (COL["R"] if j == 2 else "#999")); sp.set_linewidth(1.2)
        a.set_title(lab, fontsize=7.4, color=COL["fp16"] if j == 1 else (COL["R"] if j == 2 else "#333"))
    fig.text(0.54, 0.95, "(b) SDXL VAE decoder, same latent", fontsize=8.5)
    save(fig, "fig_teaser")


def remedies():
    """그림 2-1: fp16 오버플로 처방 비교 (한국어 라벨). 내용은 2장 본문과 같다."""
    with plt.rc_context({"font.family": KFONT}):
        cols = ["형식 승격\n(fp32 경로)", "미세조정\n(SDXL-VAE-FP16-Fix)", "고정 배수 축소\n(Draw Things)", "잔차 재조정\n(본 논문)"]
        rows = ["변경하는 것", "fp16만으로 실행", "학습 필요", "모델마다 정할 것"]
        cells = [["오버플로 연산을\nfp32로 계산", "VAE 가중치를\n다시 학습", "특정 블록 MLP를\n축소·fp32 복원,\n잔차 흐름은 fp32", "기록 층 ×c\n정규화 eps ×c²\n출력부 ×1/c"],
                 ["아니오", "예", "아니오", "예"],
                 ["아니오", "예", "아니오", "아니오"],
                 ["승격할 연산", "학습 설정", "축소할 블록과 배수", "c (진단 집합으로 결정),\n편집 층 (자동 구성 도구)"]]
        fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.set_xlim(0, 100); ax.set_ylim(5.5, 50); ax.axis("off")
        x0, cw, rh = 16, 21, 8.2
        for j, c in enumerate(cols):
            fc = "#e8eef6" if j == 3 else "#efefef"
            ax.add_patch(plt.Rectangle((x0 + j * cw, 40), cw - 0.6, 9.6, fc=fc, ec="none"))
            ax.text(x0 + j * cw + cw / 2 - 0.3, 44.8, c, ha="center", va="center", fontsize=8, fontweight="bold", color=COL["R"] if j == 3 else "#222")
        for i, r in enumerate(rows):
            hs = [11.5, 6.5, 6.5, 9.0]; h = hs[i]; y = 40 - sum(hs[:i + 1])
            ax.text(0.5, y + h / 2, r, ha="left", va="center", fontsize=8, fontweight="bold", color="#333")
            ax.plot([0, 100], [y, y], color="#cccccc", lw=0.6)
            for j in range(4):
                t = cells[i][j]
                good = (i == 1 and t == "예") or (i == 2 and t == "아니오")
                colr = (COL["R"] if j == 3 else "#222") if (i in (0, 3) or good) else "#9a9a9a"
                ax.text(x0 + j * cw + cw / 2 - 0.3, y + h / 2, t, ha="center", va="center", fontsize=7.2 if i in (0, 3) else 8, color=colr)
        save(fig, "fig_remedies")


def flow():
    """그림 3-0: 방법 전체 흐름 (한국어 라벨)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    with plt.rc_context({"font.family": KFONT}):
        fig, ax = plt.subplots(figsize=(7.2, 2.7)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
        def box(x, y, w, h, t, ec="#777", fc="white", fs=7.6, bold=False):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=1.1))
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)
        def arr(x1, y1, x2, y2):
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color="#555", lw=1.1))
        # 위 줄: 진단 → c
        box(1, 27, 20, 10, "진단 집합\n(평가와 겹치지 않는\n입력 100개)", fs=7.4)
        box(26, 27.5, 16, 9, "bf16 진단\n(연산별 최댓값)")
        box(47, 27.5, 16, 9, "오버플로 지점\n최댓값 M")
        box(68, 27.5, 16, 9, "c 결정\nM·c ≤ 32,752", ec=COL["R"])
        for a, b in ((21.8, 26), (42.8, 47), (63.8, 68)):
            arr(a, 32, b - 0.6, 32)
        # 아래 줄: 자동 구성 도구 → 확인
        ax.add_patch(FancyBboxPatch((1, 1.5), 47, 16, boxstyle="round,pad=0.5", fc="#f3f6fa", ec=COL["R"], lw=1.1))
        ax.text(24.5, 15, "자동 구성 도구", ha="center", va="center", fontsize=8, fontweight="bold", color=COL["R"])
        box(3, 3.2, 12, 7.5, "그래프\n기록"); box(18, 3.2, 12, 7.5, "영역\n탐색"); box(33, 3.2, 12, 7.5, "편집\n적용")
        arr(15.6, 7, 18, 7); arr(30.6, 7, 33, 7)
        box(54, 4, 18, 9, "fp32 등가성 확인\n(평가 입력)")
        box(77, 4, 21, 9, "fp16 실행·배포 런타임\n상실 확인 (평가 입력)", ec=COL["R"])
        arr(46.2, 8.5, 54, 8.5); arr(72.6, 8.5, 77, 8.5)
        arr(52, 27, 20, 18.6)                                # 오버플로 지점 → 도구(출발점)
        ax.text(33, 23.2, "출발점", fontsize=7, color="#444", ha="center", bbox=dict(fc="white", ec="none", pad=0.6))
        arr(74, 27, 42, 18.6)                                # c → 도구(편집)
        ax.text(60, 23.2, "c", fontsize=8, color=COL["R"], fontstyle="italic", ha="center", bbox=dict(fc="white", ec="none", pad=0.6))
        save(fig, "fig_flow")


def region():
    """그림 3-2: 자동 구성 도구의 영역 탐색 모식도 (실제 모델이 아니다)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    Y = 21
    ax.add_patch(plt.Rectangle((2, Y - 3), 82, 6, fc="#e8eef6", ec="none"))
    ax.text(2, Y + 4.2, "region: tensors scaled by c", fontsize=7.4, color=COL["R"])
    def node(x, y, t, ec="#777", w=8, h=6.4, fs=6.6):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.35", fc="white", ec=ec, lw=1.1))
        ax.text(x, y, t, ha="center", va="center", fontsize=fs)
    def arr(x1, y1, x2, y2, c="#555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=c, lw=1))
    node(6, Y, "embed\n×c (writer)", ec=COL["R"], w=9)
    node(38, Y, "add", ec=COL["R"], w=5.5, h=4.6)
    node(73, Y, "add", ec=COL["fp16"], w=5.5, h=4.6)
    ax.text(84, Y + 4.2, "start point\n(> 32,752)", ha="center", fontsize=6.6, color=COL["fp16"])
    node(93, Y, "exit\n×1/c", ec="#2f2f2f", w=7)
    arr(10.9, Y, 35, Y, COL["R"]); arr(41, Y, 70, Y, COL["R"]); arr(76, Y, 89.3, Y, COL["R"])
    for xn, xa in ((13, 38), (47, 73)):                      # 블록: 정규화 → f → 출력 투영(기록 층) → 더하기
        node(xn, 7, "norm\neps ×c²")
        node(xn + 10, 7, "block f(·)")
        node(xn + 20, 7, "proj ×c\n(writer)", ec=COL["R"])
        arr(xn - 1, Y - 3.2, xn, 10.6); arr(xn + 4.4, 7, xn + 5.6, 7); arr(xn + 14.4, 7, xn + 15.6, 7)
        arr(xn + 21, 10.6, xa - 1.5, Y - 2.6, COL["R"])
    node(56, 34, "shortcut conv\nbias only ×c", ec=COL["R"], w=12)
    arr(46, Y + 3.2, 50, 30.6); arr(62, 30.6, 71.5, Y + 2.6, COL["R"])
    ax.text(50, 0.3, "linear producer whose input is outside the region → writer (×c); input already inside → bias only (×c); "
            "consumer norm → eps ×c²; non-norm linear consumer → exit (×1/c)", ha="center", fontsize=6.0, color="#444")
    save(fig, "fig_region")


def qwen_qual():
    """그림 5-q: Qwen2.5-VL-3B 생성 답 예 — 무교정 fp16이 상실한 첫 세 문항(평가 순서 0·1·2번)."""
    import sys
    from PIL import Image
    sys.path.insert(0, str(ROOT / "scripts"))
    from dataloader import build_token_to_images                      # 평가 순서 = make_drivelm_official_lists.py와 같다
    first = {}
    for q in json.load(open(ROOT / "data/nuscenes_qa/NuScenes_val_questions.json"))["questions"]:
        first.setdefault(q["sample_token"], q["question"])
    rows = [{"image": str(c["CAM_FRONT"]), "q": first.get(t)} for t, c in
            build_token_to_images(str(ROOT / "data/QA_dataset_nus/v1_1_val_nus_q_only.json")).items() if "CAM_FRONT" in c]
    g = J("env4/dlm_gen_qwen.json"); P = g["predictions"]
    idx = [i for i, b in enumerate(g["fp16"]["per_q_bad"]) if b][:3]   # 선택 규칙: 무교정 fp16이 상실한 첫 세 문항(평가 순서)
    sel = [{**rows[i], "pred": {k: P[k][i] for k in ("fp32", "bf16", "fp16", "fp16_resid")}} for i in idx]
    fig = plt.figure(figsize=(7.2, 6.0))
    cut = lambda t, n=78: t if len(t) <= n else t[:n - 1] + "…"
    for k, r in enumerate(sel):
        y0 = 1 - (k + 1) * 0.325
        a = fig.add_axes([0.0, y0 + 0.03, 0.27, 0.27]); im = Image.open(r["image"]).convert("RGB"); im.thumbnail((640, 360))
        a.imshow(im); a.axis("off")
        t = fig.add_axes([0.29, y0 + 0.02, 0.71, 0.29]); t.axis("off"); t.set_xlim(0, 1); t.set_ylim(0, 1)
        t.text(0, 0.93, "Q: " + r["q"], fontsize=7.8, fontweight="bold", va="top")
        for j, (key, lab, c) in enumerate((("fp32", "fp32", "#2f2f2f"), ("bf16", "bf16", COL["bf16"]),
                                           ("fp16", "fp16 (uncorr.)", COL["fp16"]), ("fp16_resid", "fp16 + RR", COL["R"]))):
            yy = 0.72 - j * 0.19
            t.text(0, yy, lab, fontsize=7.2, color=c, fontweight="bold", va="top")
            t.text(0.2, yy, cut(r["pred"][key].replace("\n", " ")), fontsize=7, color="#222" if key != "fp16" else c, va="top", wrap=True)
    save(fig, "fig_qwen_qual")


def block_profile():
    """그림 3-3: Qwen2.5-VL-3B 비전 인코더 블록 출력 최댓값, 진단 집합 100장, bf16, 교정 전·잔차 재조정 뒤(개정 42 ①)."""
    d = J("env6/qwen_block_profile.json")
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for k, lab, col, mk in (("before", "uncorrected", COL["fp16"], "o"), ("after", "RR (c = 1/4)", COL["R"], "s")):
        pb = [b for b in d[k]["per_block"] if b["block"] != "tower_out"]
        ax.plot([b["block"] for b in pb], [b["max_worst"] for b in pb], marker=mk, ms=3, lw=1.2, color=col, label=lab)
    ax.axhline(65504, color=COL["fp16"], lw=1, ls="--", label="fp16 limit 65,504"); ax.axhline(32752, color="#333", lw=0.9, label="target 32,752")
    ax.set_yscale("log"); ax.set_xlabel("vision encoder block"); ax.set_ylabel("max |block output| (bf16)")
    ax.set_xlim(-0.5, 31.5)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5, ncol=2)
    save(fig, "fig_block_profile")


def flux_qual():
    """그림 5-x: FLUX.1-schnell 정성 결과 — PartiPrompts 1~4번, 원본 fp32·fp16 클리핑 끔·Draw Things 공개 설정·fp16 + RR (개정 42 ②)."""
    from PIL import Image
    D = C / "env6/flux_qual"
    d = json.load(open(D / "flux_qual.json"))
    rows = [("fp32", "original fp32", None), ("fp16_noclip", "fp16, no clipping", "fp16_noclip"),
            ("drawthings", "Draw Things", "drawthings"), ("resid", "fp16 + RR", "resid")]
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.6))
    for r, (name, lab, key) in enumerate(rows):
        for i in range(4):
            ax = axes[r][i]; ax.imshow(Image.open(D / f"{name}_{i}.png")); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(f"\u201c{d['prompts'][i]}\u201d", fontsize=8)
            if i == 0:
                ax.set_ylabel(lab, fontsize=8)
            if key:
                rw = d["rows"][key][i]
                ax.text(0.5, -0.06, "lost (NaN/Inf)" if not rw["finite"] else f"rel. err {rw['rel']:#.3g}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.18)
    save(fig, "fig_flux_qual")


if __name__ == "__main__":
    schematic(); calib_c(); fidelity(); deploy(); teaser(); remedies(); flow(); region(); qwen_qual(); block_profile(); flux_qual()
