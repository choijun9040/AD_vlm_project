"""
서론 그림 — 문제의 개요 (학위 논문 1.1절, 그림 1-1).
  (a) 배포 구성: 언어 모델은 INT4로 양자화되어 양자화 오차 지표의 검사를 받지만, 비전 인코더는 양자화에서
      제외되어 fp16으로 실행되고 그 검사를 받지 않는다. 마지막 블록 MLP의 활성이 fp16 상한을 넘으면 출력이 상실된다.
  (b) 실측 사례: 같은 이미지·같은 질문에 대한 네 조건의 출력. 문구와 수치는 손으로 적지 않고 결과 파일에서 읽는다.
데이터 (--root 기본 eval_results/full799, 원본 해상도 1,440,000):
  예측   <root>/student_{baseline_v2,full}_epoch_1_native_{bf16,fp16,fp16_fixa}_shard*of8_predictions.jsonl
  활성   <root>/res_1440000__student_{baseline_v2,full}_per_image.jsonl (마지막 블록 max, bf16)
  이미지 data/nuscenes/samples/CAM_FRONT/<image>
실행:
    python scripts/plot_intro_overview.py                 # 2단 폭
    python scripts/plot_intro_overview.py --layout single # 1단 폭
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e4e1"
BAD, GOOD, NEUTRAL = "#eb6834", "#2a78d6", "#8a918c"   # 참조 팔레트 slot 2·1, 중립
FP16_MAX = 65504

# 사례 — 기준 모델이 bf16에서 맞히고 fp16에서 상실하며, 교정 fp16과 Full fp16이 다시 맞힌 버스 상태 문항
IMAGE = "n015-2018-07-18-11-41-49+0800__CAM_FRONT__1531885354662530.jpg"
QUESTION = "What is the status of the bus?"

LAYOUT = {
    "double": dict(figsize=(7.2, 3.4), base_fs=8.5, small_fs=7.2, rows=1),
    "single": dict(figsize=(3.5, 6.2), base_fs=7.5, small_fs=6.4, rows=2),
}


def pred(root, tag):
    for f in sorted(glob.glob(f"{root}/{tag}_shard*of8_predictions.jsonl")):
        for ln in open(f):
            r = json.loads(ln)
            if r["image"] == IMAGE and r["question"] == QUESTION:
                return r
    raise SystemExit(f"{tag}에 사례 문항이 없다")


def act(root, model):
    for ln in open(f"{root}/res_1440000__{model}_per_image.jsonl"):
        r = json.loads(ln)
        if r["image"] == IMAGE and r["dtype"] == "bfloat16":
            return r["blk31"]
    raise SystemExit(f"{model}에 사례 이미지가 없다")


def box(ax, x, y, w, h, text, fc, ec, L, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=L["small_fs"], color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.0))


def panel_a(ax, L):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("(a) common deployment setup", fontsize=L["base_fs"], color=INK, pad=6)
    # 위에서 아래로: 이미지 → 비전 인코더(fp16) → 언어 모델(INT4). 설명은 오른쪽에
    box(ax, 0.04, 0.80, 0.42, 0.12, "camera image", "white", NEUTRAL, L)
    box(ax, 0.04, 0.47, 0.42, 0.20, "vision encoder\nfp16", "#fde8de", BAD, L)
    box(ax, 0.04, 0.14, 0.42, 0.20, "language model\nINT4", "#e3eefb", GOOD, L)
    arrow(ax, 0.25, 0.80, 0.25, 0.67); arrow(ax, 0.25, 0.47, 0.25, 0.34)
    ax.text(0.52, 0.66, "excluded from quantization,\nnot checked by its metrics", ha="left", va="top",
            fontsize=L["small_fs"], color=BAD, linespacing=1.2)
    ax.text(0.52, 0.52, "last-block activation\n> 65,504 (fp16 limit)\n→ inf → NaN\n→ whole output lost",
            ha="left", va="top", fontsize=L["small_fs"], color=INK, linespacing=1.2)
    ax.text(0.52, 0.30, "checked by\nquantization SNR,\noutlier ratio", ha="left", va="top",
            fontsize=L["small_fs"], color=GOOD, linespacing=1.2)


def panel_b(ax_img, ax_txt, rows, acts, L):
    im = Image.open(f"data/nuscenes/samples/CAM_FRONT/{IMAGE}")
    ax_img.imshow(im); ax_img.axis("off")
    ax_img.set_title("(b) same image, same question", fontsize=L["base_fs"], color=INK, pad=6)
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1); ax_txt.axis("off")
    ax_txt.text(0.0, 0.97, f"Q: {QUESTION}   (answer: {rows[0][1]['gt']})", fontsize=L["small_fs"],
                color=INK, va="top")
    y = 0.80
    for label, r, note in rows:
        out = r["pred_raw"].replace("<|im_end|>", "")
        out = out if len(out) <= 18 else out[:16] + "…"
        ok = r["correct"]
        ax_txt.text(0.0, y, label, fontsize=L["small_fs"], color=INK2, va="top")
        ax_txt.text(0.56, y, f"\"{out}\"", fontsize=L["small_fs"], color=GOOD if ok else BAD, va="top",
                    family="monospace")
        ax_txt.text(0.56, y - 0.10, note, fontsize=L["small_fs"] - 0.6, color=INK2, va="top")
        y -= 0.22
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="double", choices=list(LAYOUT))
    ap.add_argument("--out", default="figures/fig0_intro_overview")
    ap.add_argument("--root", default="eval_results/full799")
    args = ap.parse_args()
    L = LAYOUT[args.layout]

    t = "student_baseline_v2_epoch_1_native"
    b16, f16, fix = pred(args.root, f"{t}_bf16"), pred(args.root, f"{t}_fp16"), pred(args.root, f"{t}_fp16_fixa")
    full = pred(args.root, "student_full_epoch_1_native_fp16")
    a_base, a_full = act(args.root, "student_baseline_v2"), act(args.root, "student_full")
    rows = [("Task only, bf16", b16, f"max activation {a_base:,.0f}"),
            ("Task only, fp16", f16, f"{a_base:,.0f} > {FP16_MAX:,}: output lost"),
            ("Task only, fp16 + correction", fix, "weights rescaled, no training"),
            ("Full (L_align + L_atc), fp16", full, f"max activation {a_full:,.0f}")]
    assert b16["correct"] and not f16["correct"] and fix["correct"] and full["correct"], "사례 조건이 맞지 않는다"

    plt.rcParams.update({"font.size": L["base_fs"]})
    fig = plt.figure(figsize=L["figsize"])
    if L["rows"] == 1:
        gs = fig.add_gridspec(2, 2, width_ratios=[0.85, 1.25], height_ratios=[1.25, 1.0], wspace=0.08, hspace=0.05)
        panel_a(fig.add_subplot(gs[:, 0]), L)
        panel_b(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), rows, None, L)
    else:
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.9, 0.9], hspace=0.12)
        panel_a(fig.add_subplot(gs[0]), L)
        panel_b(fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), rows, None, L)
    out = args.out + ("_single" if args.layout == "single" else "")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"→ {out}.png/.pdf")
    for label, r, note in rows:
        print(f"  {label:32s} {r['pred_raw'][:20]!r:24s} 정답 {r['correct']}  {note}")


if __name__ == "__main__":
    main()
