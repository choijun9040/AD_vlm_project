"""
INT8 활성 양자화 오차 프로파일
================================
"fp16으로 내보낸 게 잘못이지, 학습 손실 탓이 아니다"라는 반론에 답하기 위한 측정.

fp16 붕괴는 **절대 크기**가 표현 한계를 넘어서 생긴다. 반면 INT8 활성 양자화는
스케일이 절대 크기를 흡수하므로, 오차를 좌우하는 것은 크기가 아니라
**max 대비 전형값의 비(이상치 비율)** 다. 따라서 둘은 다른 양이며, 손실 설계가
전자만 바꾸고 후자는 그대로일 수도 있다. 그 경우도 그대로 보고해야 하므로
두 지표를 함께 잰다:

  - max|x|, rms(x), max/rms          → 이상치 비율
  - per-tensor  INT8 대칭 양자화 SNR  → 실제 배포에서 흔한 W8A8 경로
  - per-channel INT8 대칭 양자화 SNR  → 채널별 스케일로 이상치를 흡수했을 때

실행:
    python scripts/profile_int8_quant_error.py --limit 200
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images
from profile_vision_activations import (
    DRIVELM_VAL_JSON, BASE_3B, BASE_7B, MAX_PIXELS, MIN_PIXELS,
    build_pixel_cache, get_visual, percentile,
)

# 재는 층: align이 작용하는 구간(16/23/24/28)과 fp16 한계에 걸리는 마지막 블록(31)
LAYERS = (16, 23, 24, 28, 31)

MODELS = [
    ("base_3b",             BASE_3B, None,                                      "미세조정 없음"),
    ("student_baseline_v2", BASE_3B, "checkpoints/student_baseline_v2/epoch_1", "task CE only"),
    ("student_kd_only_v4",  BASE_3B, "checkpoints/student_kd_only_v4/epoch_1",  "task + 출력 KD"),
    ("student_temporal",    BASE_3B, "checkpoints/student_temporal/epoch_1",    "task + atc"),
    ("student_spatial",     BASE_3B, "checkpoints/student_spatial/epoch_1",     "task + align"),
    ("student_full",        BASE_3B, "checkpoints/student_full/epoch_1",        "task + align + atc"),
]


def int8_snr(x: torch.Tensor, per_channel: bool) -> float:
    """INT8 대칭 양자화 후 신호 대 잡음비(dB). 높을수록 양자화에 강건하다."""
    x = x.float()
    if per_channel:                       # 마지막 차원(hidden)마다 스케일
        amax = x.abs().amax(dim=tuple(range(x.dim() - 1)), keepdim=True)
    else:                                 # 텐서 전체 하나의 스케일
        amax = x.abs().max()
    scale = (amax / 127.0).clamp_min(1e-12)
    xq = torch.clamp(torch.round(x / scale), -127, 127) * scale
    sig = x.pow(2).sum()
    err = (x - xq).pow(2).sum()
    if err <= 0:
        return float("inf")
    return float(10.0 * torch.log10(sig / err))


@torch.no_grad()
def profile(name, base_id, adapter, pixel_cache, log_every):
    print(f"\n{'=' * 70}\n[{name}] adapter={adapter}\n{'=' * 70}", flush=True)
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to("cuda").eval()
    visual = get_visual(model)
    print(f"  로드 {time.time() - t0:.0f}s", flush=True)

    rec = {}

    def hook(i):
        def fn(_m, _inp, out):
            rec[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    handles = [visual.blocks[i].register_forward_hook(hook(i)) for i in LAYERS]

    acc = {i: {"max": [], "rms": [], "ratio": [], "snr_tensor": [], "snr_channel": []}
           for i in LAYERS}

    t1 = time.time()
    for n, (pv_cpu, grid_cpu) in enumerate(pixel_cache):
        rec.clear()
        visual(pv_cpu.to("cuda", torch.bfloat16, non_blocking=True),
               grid_cpu.to("cuda", non_blocking=True))
        for i in LAYERS:
            x = rec[i].float()
            mx = x.abs().max().item()
            rms = x.pow(2).mean().sqrt().item()
            acc[i]["max"].append(mx)
            acc[i]["rms"].append(rms)
            acc[i]["ratio"].append(mx / max(rms, 1e-12))
            acc[i]["snr_tensor"].append(int8_snr(x, per_channel=False))
            acc[i]["snr_channel"].append(int8_snr(x, per_channel=True))
        if log_every and (n + 1) % log_every == 0:
            print(f"    {n + 1}/{len(pixel_cache)}", flush=True)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    def summ(v):
        sv = sorted(x for x in v if math.isfinite(x))
        return {"p50": percentile(sv, 0.50), "p95": percentile(sv, 0.95),
                "min": sv[0] if sv else None, "max": sv[-1] if sv else None}

    out = {str(i): {k: summ(v) for k, v in acc[i].items()} for i in LAYERS}
    print(f"  완료 {time.time() - t1:.0f}s", flush=True)
    for i in LAYERS:
        s = out[str(i)]
        print(f"    blk{i:<2d} max={s['max']['p50']:>10,.0f}  rms={s['rms']['p50']:>8.1f}  "
              f"max/rms={s['ratio']['p50']:>7.1f}  "
              f"SNR(tensor)={s['snr_tensor']['p50']:>6.2f}dB  "
              f"SNR(chan)={s['snr_channel']['p50']:>6.2f}dB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="eval_results/int8_quant_error.json")
    # 2026-09-15 추가. 초판은 평가 해상도(200,704)에서만 쟀는데, 여유·붕괴율은
    # 원본 해상도(1,440,000)에서 쟀다. 그 둘을 한 표에 실으면 **조건이 다른 열이
    # 섞인다** — 지표 역전 주장이 걸린 표이므로 같은 조건에서 다시 잰다.
    ap.add_argument("--max_pixels", type=int, default=None,
                    help="미지정이면 평가 해상도 기본값(profile_vision_activations.MAX_PIXELS)")
    ap.add_argument("--min_pixels", type=int, default=None)
    args = ap.parse_args()

    awq_compat.patch()

    token_to_images = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in token_to_images.values() if "CAM_FRONT" in v][: args.limit]
    print(f"대상 이미지: {len(paths)}장")

    mxp = args.max_pixels or MAX_PIXELS
    mnp = args.min_pixels if args.min_pixels is not None else MIN_PIXELS
    print(f"[해상도] max_pixels={mxp:,}  min_pixels={mnp:,}")
    processor = AutoProcessor.from_pretrained(BASE_3B, max_pixels=mxp, min_pixels=mnp)
    cache = build_pixel_cache(processor, paths, args.workers)

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["_meta"] = {"n_images": len(paths), "layers": list(LAYERS),
                        "max_pixels": mxp, "min_pixels": mnp}

    for name, base_id, adapter, note in MODELS:
        r = profile(name, base_id, adapter, cache, args.log_every)
        results[name] = {"loss": note, "adapter": adapter, "by_layer": r}
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"  저장: {out_path}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'모델':22s}{'손실':20s}{'blk31 max/rms':>15}{'SNR/tensor':>12}{'SNR/chan':>11}")
    print("-" * 92)
    for name, *_ in MODELS:
        if name not in results:
            continue
        s = results[name]["by_layer"]["31"]
        print(f"{name:22s}{results[name]['loss']:20s}{s['ratio']['p50']:>15.1f}"
              f"{s['snr_tensor']['p50']:>11.2f}dB{s['snr_channel']['p50']:>10.2f}dB")
    print("=" * 92)
    print("해석: max/rms가 클수록, SNR이 낮을수록 INT8 활성 양자화에 취약하다.")


if __name__ == "__main__":
    main()
