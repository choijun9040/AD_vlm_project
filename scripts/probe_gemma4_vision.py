"""
Gemma 4 비전 인코더의 fp16 여유·상실·교정 — 범위 확장 A (사전 등록: thesis_outline §6 「범위 확장 A·B」, 2026-10-20).

vLLM #40290이 보고한 Gemma 4(31B·26B-A4B, `standardize=True`)의 fp16 출력 상실을 본 연구의 규칙으로 잰다.
넘칠 수 있는 곳은 블록 **뒤**다 — pooler의 `× √hidden(≈33.9)`와 표준화 `(x − std_bias) × std_scale`의 뺄셈.
transformers 5.20의 기준 구현은 이 두 연산을 float32로 올려 계산하므로, 상실은 그 승격이 없는 경로에서 생긴다.

측정 (headroom_guard와 같은 규칙):
  1. 크기 — bf16(블록)·fp32(블록 뒤)에서 층별·지점별 이미지당 max|·|
  2. 상실 — fp16 두 경로: (i) 기준 구현 그대로 (ii) 블록 뒤 두 연산까지 fp16
  3. 교정 — 잔차 흐름 재조정(가중치 편집만). 잔차에 쓰는 세 곳(`input_proj`·위치 임베딩 표,
     층마다 `post_attention_layernorm`·`post_feedforward_layernorm`)에 c를 곱하면 잔차 전체가 정확히 c배가 되고
     (층 입력 정규화는 RMSNorm `x·w`라 스케일 불변), `std_bias ← c·std_bias`, `std_scale ← std_scale/c`로
     최종 출력이 원래와 같아진다. c는 도구 규칙대로 최악 여유 2.0을 맞추는 값.

    python scripts/probe_gemma4_vision.py --model_dir /workspace/gemma4_vision/gemma-4-26B-A4B-it --max_soft_tokens 280
"""

import argparse
import copy
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

sys.path.append(str(Path(__file__).parent))
from dataloader import build_token_to_images

DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
FP16_MAX = 65504.0
TARGET = 2.0                     # headroom_guard --target_headroom 기본값 (설계 판단)


def load_vision(model_dir, dtype):
    from transformers import Gemma4Config
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel
    cfg = Gemma4Config.from_pretrained(model_dir)
    vm = Gemma4VisionModel(cfg.vision_config)
    sd = load_file(str(Path(model_dir) / "vision.safetensors"))
    sd = {k[len("model.vision_tower."):]: v for k, v in sd.items() if k.startswith("model.vision_tower.")}
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"가중치 불일치 — 빠짐 {missing[:3]} / 남음 {unexpected[:3]}")
    return vm.eval().to("cuda", getattr(torch, dtype))


def build_inputs(model_dir, max_soft_tokens, limit, offset=0):
    from transformers import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained(model_dir, max_soft_tokens=max_soft_tokens)
    tok = build_token_to_images(DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][offset:offset + limit]
    cache = []
    for p in paths:
        enc = ip(images=[Image.open(p).convert("RGB")], return_tensors="pt")
        cache.append((enc["pixel_values"], enc["image_position_ids"]))
    return paths, cache


def forward_points(vm, pv, pos, upcast):
    """기준 구현의 forward를 그대로 따르되 지점별 값을 돌려준다.

    upcast=True  → 기준 구현과 같다 (pooler 스케일·표준화를 float32에서)
    upcast=False → 경로 (ii): 두 연산을 모델 dtype에서 (vLLM 수정 전·fp16 엔진 모사)
    """
    dt = next(vm.parameters()).dtype
    pv, pos = pv.to("cuda", dt), pos.to("cuda")
    k = vm.config.pooling_kernel_size
    L = pv.shape[-2] // (k * k)
    pad = (pos == -1).all(dim=-1)
    x = vm.patch_embedder(pv, pos, pad)
    layer_out = []
    hs = [lyr.register_forward_hook(lambda _m, _i, o: layer_out.append(o[0] if isinstance(o, tuple) else o))
          for lyr in vm.encoder.layers]
    h = vm.encoder(inputs_embeds=x, attention_mask=~pad, pixel_position_ids=pos).last_hidden_state
    for hk in hs:
        hk.remove()
    h = h.masked_fill(pad.unsqueeze(-1), 0.0)
    h, mask = vm.pooler._avg_pool_by_positions(h, pos, L)          # 평균 — 크기를 키우지 않는다
    if upcast:
        h = h.float()
    pooled = h * vm.pooler.root_hidden_size
    pooled = pooled[mask]
    sub = pooled - (vm.std_bias.float() if upcast else vm.std_bias)
    final = sub * (vm.std_scale.float() if upcast else vm.std_scale)
    return {"layers": layer_out, "pooled_scaled": pooled, "minus_bias": sub, "final": final.to(dt)}


def amax(t):
    t = t.detach().float()
    return float("inf") if not torch.isfinite(t).all() else float(t.abs().max())


@torch.no_grad()
def profile(vm, cache, upcast):
    names = [f"L{i:02d}" for i in range(len(vm.encoder.layers))] + ["pooled_scaled", "minus_bias", "final"]
    rec = {n: [] for n in names}
    finals = []
    for pv, pos in cache:
        r = forward_points(vm, pv, pos, upcast)
        for i, t in enumerate(r["layers"]):
            rec[f"L{i:02d}"].append(amax(t))
        for n in ("pooled_scaled", "minus_bias", "final"):
            rec[n].append(amax(r[n]))
        finals.append(r["final"].float().cpu())
    return rec, finals


def summarize(rec):
    rows = []
    for n, vals in rec.items():
        fin = [v for v in vals if math.isfinite(v)]
        p50 = st.median(fin) if fin else float("nan")
        worst = max(fin) if fin else float("nan")
        rows.append({"point": n, "max_p50": p50, "max_worst": worst,
                     "headroom_p50": FP16_MAX / p50 if p50 else None,
                     "headroom_worst": FP16_MAX / worst if worst else None,
                     "exceed_rate": sum(1 for v in vals if not math.isfinite(v) or v > FP16_MAX) / len(vals)})
    return rows


def collapse_rate(finals):
    return sum(1 for f in finals if not torch.isfinite(f).all()) / len(finals)


@torch.no_grad()
def apply_residual_rescale(vm, c):
    """잔차 흐름 전체를 c배로 — 가중치 편집만 (docstring 참조)."""
    vm = copy.deepcopy(vm)
    vm.patch_embedder.input_proj.weight.mul_(c)
    vm.patch_embedder.position_embedding_table.mul_(c)
    for lyr in vm.encoder.layers:
        lyr.post_attention_layernorm.weight.mul_(c)
        lyr.post_feedforward_layernorm.weight.mul_(c)
    vm.std_bias.mul_(c)
    vm.std_scale.div_(c)
    return vm


def cosines(a, b):
    out = []
    for x, y in zip(a, b):
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            out.append(float("nan")); continue
        out.append(float(F.cosine_similarity(x.flatten().double(), y.flatten().double(), dim=0)))
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--max_soft_tokens", type=int, default=280)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0,
                    help="이미지 시작 위치. A1 확인 실험은 100(첫 실험의 100장과 겹치지 않게)")
    ap.add_argument("--out_dir", default="eval_results")
    args = ap.parse_args()
    tag = f"{Path(args.model_dir).name}_st{args.max_soft_tokens}" + (f"_off{args.offset}" if args.offset else "")

    paths, cache = build_inputs(args.model_dir, args.max_soft_tokens, args.limit, args.offset)
    print(f"[설정] {tag}  이미지 {len(paths)}장  pixel_values={tuple(cache[0][0].shape)}")

    # 1) 크기 — bf16 블록, 블록 뒤는 기준 구현대로 fp32
    vb = load_vision(args.model_dir, "bfloat16")
    rec_b, fin_b = profile(vb, cache, upcast=True)
    mag = summarize(rec_b)
    # 2) 상실 — fp16 두 경로
    vh = load_vision(args.model_dir, "float16")
    _, fin_i = profile(vh, cache, upcast=True)
    rec_ii, fin_ii = profile(vh, cache, upcast=False)
    worst_row = min((r for r in mag if r["headroom_worst"]), key=lambda r: r["headroom_worst"])
    exceed_any = [max(1 if (not math.isfinite(rec_b[n][j]) or rec_b[n][j] > FP16_MAX) else 0 for n in rec_b)
                  for j in range(len(paths))]
    pred = sum(exceed_any) / len(paths)
    print(f"[크기]  최악 여유 {worst_row['headroom_worst']:.3f}배 @ {worst_row['point']}  "
          f"초과 비율(어느 지점이든) {pred:.1%}")
    print(f"[상실]  (i) 기준 구현 {collapse_rate(fin_i):.1%}   (ii) 승격 없음 {collapse_rate(fin_ii):.1%}")

    # 3) 교정 — 잔차 재조정
    M = max(max(v for v in vals if math.isfinite(v)) for vals in rec_b.values())
    # c는 목표 여유를 넘기는 **가장 큰 2의 거듭제곱**으로 정한다 — 가중치에 곱할 때 지수만 바뀌어
    # bf16·fp16 반올림이 생기지 않으므로, 등가성 검사가 반올림이 아니라 교정 원리만 본다
    # (결과 보기 전 결정, 2026-10-20). 최악 여유 ≥ 2.0이라는 사전 등록 기준은 그대로 만족한다.
    c = min(1.0, 2.0 ** math.floor(math.log2(FP16_MAX / (TARGET * M))))
    vb_c = apply_residual_rescale(vb, c)
    rec_bc, fin_bc = profile(vb_c, cache, upcast=True)
    cos_eq = cosines(fin_b, fin_bc)
    vh_c = apply_residual_rescale(vh, c)
    rec_iic, fin_iic = profile(vh_c, cache, upcast=False)
    cos_ii = cosines(fin_b, fin_iic)
    fin_cos = [x for x in cos_ii if math.isfinite(x)]

    # **사후 분석 (2026-10-20, 2장 시험에서 bf16 등가성이 0.998로 사전 등록 기준 0.9999에 미달한 뒤 추가).**
    # 사전 등록 기준(bf16 전후 코사인)은 위에서 그대로 재고 그대로 보고한다. 아래는 미달의 원인을 읽기 위한 것이다 —
    # RMSNorm의 fp32 `pow(ms, −0.5)`가 c배 입력에서 마지막 자리가 달라져 bf16 반올림 경계의 원소를 한 칸 뒤집는다.
    v32 = load_vision(args.model_dir, "float32")
    _, fin_32 = profile(v32, cache, upcast=True)
    _, fin_32c = profile(apply_residual_rescale(v32, c), cache, upcast=True)
    del v32
    post = {"equivalence_cos_min_fp32": min(cosines(fin_32, fin_32c)),
            "noise_floor_cos_min_bf16_vs_fp32": min(cosines(fin_32, fin_b)),
            "deploy_cos_min_fp16_i_reference_vs_fp32": min(x for x in cosines(fin_32, fin_i) if math.isfinite(x)),
            "deploy_cos_min_fp16_ii_fixed_vs_fp32": min(x for x in cosines(fin_32, fin_iic) if math.isfinite(x))}
    print(f"[사후]  fp32 등가성 {post['equivalence_cos_min_fp32']:.6f}  bf16 잡음 하한 {post['noise_floor_cos_min_bf16_vs_fp32']:.6f}  "
          f"fp16(i) 기준 구현 {post['deploy_cos_min_fp16_i_reference_vs_fp32']:.6f}  "
          f"fp16(ii) 교정 {post['deploy_cos_min_fp16_ii_fixed_vs_fp32']:.6f}  (모두 fp32 원본 대비 최소 코사인)")
    print(f"[교정]  c={c:.4f} (최대 크기 {M:,.0f} → 목표 여유 {TARGET})")
    print(f"        등가성(bf16 전후 코사인) 최소 {min(cos_eq):.6f}   "
          f"(ii) 교정 후 상실 {collapse_rate(fin_iic):.1%}   bf16 대비 코사인 최소 "
          f"{min(fin_cos) if fin_cos else float('nan'):.6f}")

    out = {"tag": tag, "model_dir": args.model_dir, "max_soft_tokens": args.max_soft_tokens, "offset": args.offset,
           "n_images": len(paths), "pixel_values_shape": list(cache[0][0].shape),
           "magnitude": mag, "predicted_exceed_rate": pred,
           "collapse_i_reference": collapse_rate(fin_i), "collapse_ii_no_upcast": collapse_rate(fin_ii),
           "fp16_ii_per_point_exceed": {n: sum(1 for v in vals if not math.isfinite(v)) / len(vals)
                                        for n, vals in rec_ii.items()},
           "fix": {"c": c, "max_magnitude": M, "target_headroom": TARGET,
                   "equivalence_cos_min_bf16": min(cos_eq),
                   "collapse_ii_after_fix": collapse_rate(fin_iic),
                   "cos_vs_bf16_min_after_fix_ii": min(fin_cos) if fin_cos else None,
                   "magnitude_after_fix": summarize(rec_bc)},
           "post_hoc": post}
    p = Path(args.out_dir) / f"gemma4_vision_{tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {p}")


if __name__ == "__main__":
    main()
