"""
849번 채널의 정체를 규명한다 — 5장 메커니즘의 전제 검증 (방법 1·2).

**배경.** `weight_distribution.json`에서 `down_proj` 849행이 blk23(2.42배)과 blk31(9.33배)
**양쪽에서 최상위**로 나왔다. `down_proj`의 출력 차원은 1280 = 잔차 스트림 폭이고,
`gate/up_proj`(3420 = MLP 내부 폭)에는 849가 나타나지 않는다. 즉 849는 특정 층의
우연한 행이 아니라 **잔차 스트림의 한 채널**일 가능성이 높다.

**방법 1 (깊이 프로파일)** — 32개 블록 전부에서 849행 노름과 그 순위를 잰다.
단조 증가하면 "전역 잔차 채널이 깊이에 따라 증폭된다"가 데이터로 확정된다.
가중치만 보므로 순전파가 필요 없다.

**방법 2 (활성 대응)** — blk31 출력에서 **실제로 큰 값이 나오는 채널**이 849인지 본다.
지금 논문은 "849행 이상치가 증폭기"라고 쓰는데, 오버플로가 다른 채널에서 일어나면
그 서술이 틀린다. **이 검증 없이는 5장을 쓸 수 없다.**

실행:
    python scripts/probe_channel849.py --limit 50
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/student_baseline_v2/epoch_1",
                    help="'none'이면 사전학습 베이스")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--channel", type=int, default=849)
    ap.add_argument("--out", default="eval_results/channel849.json")
    args = ap.parse_args()

    awq_compat.patch()
    print(f"[로드] {BASE} (+ {args.checkpoint})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32, attn_implementation="eager")
    if args.checkpoint.lower() != "none":
        model = PeftModel.from_pretrained(model, args.checkpoint).merge_and_unload()
    tower = model.visual.eval().to("cuda")
    C = args.channel

    # ---------------- 방법 1: 깊이 프로파일 (가중치만) ----------------
    print(f"\n=== 방법 1 — {C}행의 깊이 프로파일 ({len(tower.blocks)}개 블록) ===")
    print(f"{'blk':>4}{'849 노름':>12}{'중앙값':>10}{'배율':>8}{'순위':>7}{'최상위 행':>10}")
    depth_rows = []
    for i, blk in enumerate(tower.blocks):
        w = blk.mlp.down_proj.weight.detach().float()      # (1280, 3420)
        n = torch.linalg.norm(w, dim=1)
        med = float(n.median())
        val = float(n[C])
        rank = int((n > n[C]).sum()) + 1
        top = int(n.argmax())
        depth_rows.append({"block": i, "norm": val, "median": med,
                           "ratio": val / med, "rank": rank, "top_row": top,
                           "top_ratio": float(n.max()) / med})
        if i % 4 == 0 or i >= len(tower.blocks) - 4:
            print(f"{i:>4}{val:>12.3f}{med:>10.3f}{val/med:>8.2f}{rank:>7}{top:>10}")

    top_is_C = [r["block"] for r in depth_rows if r["top_row"] == C]
    print(f"\n  {C}행이 최상위인 블록: {len(top_is_C)}/{len(depth_rows)}개 — {top_is_C}")
    ratios = [r["ratio"] for r in depth_rows]
    print(f"  배율 범위: {min(ratios):.2f} (blk{int(np.argmin(ratios))}) "
          f"→ {max(ratios):.2f} (blk{int(np.argmax(ratios))})")
    inc = sum(1 for a, b in zip(ratios, ratios[1:]) if b >= a)
    print(f"  깊이에 따라 증가한 구간: {inc}/{len(ratios)-1}")

    # ---------------- 방법 2: 활성 채널 대응 ----------------
    print(f"\n=== 방법 2 — 마지막 블록 출력에서 실제로 큰 채널은 어디인가 ===")
    proc = AutoProcessor.from_pretrained(BASE, max_pixels=args.max_pixels,
                                         min_pixels=args.min_pixels)
    tok = build_token_to_images(VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:args.limit]
    print(f"  이미지 {len(paths)}장, max_pixels={args.max_pixels}")

    captured = {}
    def hook(m, i, o):
        captured["mlp"] = o.detach()
    h = tower.blocks[-1].mlp.register_forward_hook(hook)

    argmax_ch, ch_share, c_absmax, glob_absmax = Counter(), [], [], []
    with torch.no_grad():
        for p in paths:
            enc = proc.image_processor(images=[Image.open(p).convert("RGB")],
                                       return_tensors="pt")
            tower(enc["pixel_values"].to("cuda", torch.float32),
                  enc["image_grid_thw"].to("cuda"))
            a = captured["mlp"].abs()                      # (seq, 1280)
            per_ch = a.max(dim=0).values                   # 채널별 최대
            argmax_ch[int(per_ch.argmax())] += 1
            g = float(per_ch.max())
            c = float(per_ch[C])
            glob_absmax.append(g); c_absmax.append(c); ch_share.append(c / g)
    h.remove()

    top_ch = argmax_ch.most_common(5)
    print(f"  최대 활성 채널 빈도 (상위 5): {top_ch}")
    print(f"  {C}번 채널이 최대였던 이미지: {argmax_ch[C]}/{len(paths)}장")
    print(f"  전역 최대 |활성| 중앙값 = {np.median(glob_absmax):,.0f}")
    print(f"  {C}번 채널 |활성| 중앙값 = {np.median(c_absmax):,.0f}")
    print(f"  {C}번 / 전역 최대 비율 중앙값 = {np.median(ch_share):.4f}")

    if argmax_ch[C] / len(paths) >= 0.9:
        verdict = f"활성 최대 채널이 거의 항상 {C}번 — 가중치 이상치와 활성 이상치가 일치한다"
    elif np.median(ch_share) >= 0.9:
        verdict = f"{C}번이 항상 최대는 아니나 전역 최대의 90% 이상 — 사실상 같은 채널대"
    else:
        verdict = (f"**불일치** — 활성 최대는 주로 {top_ch[0][0]}번 채널이다. "
                   f"'{C}행 이상치가 증폭기'라는 서술을 고쳐야 한다")
    print(f"\n  판정: {verdict}")

    res = {"checkpoint": args.checkpoint, "channel": C, "n_images": len(paths),
           "max_pixels": args.max_pixels,
           "depth_profile": depth_rows,
           "blocks_where_top": top_is_C,
           "activation": {
               "argmax_channel_top5": top_ch,
               "n_images_where_C_is_argmax": argmax_ch[C],
               "global_absmax_p50": float(np.median(glob_absmax)),
               "channel_absmax_p50": float(np.median(c_absmax)),
               "share_p50": float(np.median(ch_share))},
           "verdict": verdict}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
