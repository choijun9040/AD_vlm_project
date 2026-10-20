"""
언어 모델 백본의 fp16 여유 측정 — 범위 확장 B (사전 등록: thesis_outline §6 「범위 확장 A·B」, 2026-10-20).

지금까지는 비전 인코더만 쟀다. 실무 구성(백본 4비트·비전 인코더 fp16)에서도 백본의 **활성**은 fp16으로
흐르므로, 디코더 층의 잔차·MLP 출력·어텐션 출력이 fp16 상한(65,504)에 얼마나 가까운지 잰다.

측정 규칙은 headroom_guard와 같다 — **크기는 넘치지 않는 형식(bf16)에서, 상실은 목표 형식(fp16)에서.**
상실 측정에서는 언어 모델만 fp16으로 두고 비전 인코더는 bf16으로 둔다(비전 쪽 상실과 분리하기 위해).

    python scripts/probe_llm_headroom.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --tag baseline_v2
    python scripts/probe_llm_headroom.py --checkpoint none --tag pretrained
"""

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from dataloader import build_token_to_images

BASE_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
NUSCENESQA_VAL = "data/nuscenes_qa/NuScenes_val_questions.json"
FP16_MAX = 65504.0
NATIVE_MAX_PIXELS = 1_440_000          # 원본 해상도 (headroom_guard 표 30·E1과 같은 값)
MIN_PIXELS = 64 * 28 * 28              # 평가 스크립트와 같은 하한
IMAGE_TOKEN_ID = 151655


def load(checkpoint, dtype):
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_3B, torch_dtype=getattr(torch, dtype), attn_implementation="eager")
    if checkpoint and checkpoint.lower() != "none":
        m = PeftModel.from_pretrained(m, checkpoint).merge_and_unload()
    m = m.eval().to("cuda")
    # 비전 인코더는 언제나 bf16 — 백본의 상실만 보기 위해서다.
    m.visual.to(torch.bfloat16)
    return m


def pick_inputs(limit):
    """headroom_guard와 같은 이미지 순서(DriveLM 검증 키프레임 순), 이미지마다 첫 NuScenes-QA 문항."""
    tok = build_token_to_images(DRIVELM_VAL)
    qs = json.load(open(NUSCENESQA_VAL))["questions"]
    first_q = {}
    for q in qs:
        first_q.setdefault(q["sample_token"], q["question"])
    items = []
    for t, cams in tok.items():
        if "CAM_FRONT" in cams and t in first_q:
            items.append((t, str(cams["CAM_FRONT"]), first_q[t]))
        if len(items) >= limit:
            break
    return items


def hook_points(model):
    lm = model.model
    pts = {}
    for i, layer in enumerate(lm.layers):
        pts[f"L{i:02d}.out"] = layer
        pts[f"L{i:02d}.down_proj"] = layer.mlp.down_proj
        pts[f"L{i:02d}.o_proj"] = layer.self_attn.o_proj
    pts["final_norm"] = lm.norm
    return pts


@torch.no_grad()
def run(model, processor, items, record_pos):
    pts = hook_points(model)
    rec = {k: [] for k in pts}
    where = {k: [] for k in pts}
    cur = {}

    def mk(name):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            a = t.detach().float().abs()
            if not torch.isfinite(a).all():
                rec[name].append(float("inf")); where[name].append("nonfinite"); return
            flat = a.reshape(-1, a.shape[-1]).max(dim=-1).values      # 위치별 max
            j = int(flat.argmax())
            rec[name].append(float(flat[j]))
            if record_pos:
                ids = cur["ids"]
                kind = "first" if j == 0 else ("image" if int(ids[j]) == IMAGE_TOKEN_ID else "text")
                where[name].append(kind)
        return fn

    hs = [m.register_forward_hook(mk(n)) for n, m in pts.items()]
    nonfinite_logits = 0
    for _tok, path, q in items:
        msgs = [{"role": "user", "content": [{"type": "image", "image": path},
                                             {"type": "text", "text": q}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=[Image.open(path).convert("RGB")],
                        return_tensors="pt").to("cuda")
        cur["ids"] = inp["input_ids"][0]
        out = model(**inp)
        if not torch.isfinite(out.logits).all():
            nonfinite_logits += 1
    for h in hs:
        h.remove()
    return rec, where, nonfinite_logits / len(items)


def summarize(rec, where):
    rows = []
    for k, vals in rec.items():
        fin = [v for v in vals if math.isfinite(v)]
        p50 = st.median(fin) if fin else float("nan")
        worst = max(fin) if fin else float("nan")
        exceed = sum(1 for v in vals if (not math.isfinite(v)) or v > FP16_MAX) / len(vals)
        kinds = {}
        for w in where.get(k, []):
            kinds[w] = kinds.get(w, 0) + 1
        rows.append({"point": k, "max_p50": p50, "max_worst": worst,
                     "headroom_p50": FP16_MAX / p50 if p50 else None,
                     "headroom_worst": FP16_MAX / worst if worst else None,
                     "exceed_rate": exceed, "argmax_kind": kinds})
    return rows


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--checkpoint", default="none", help="'none'이면 사전학습 3B")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_pixels", type=int, default=NATIVE_MAX_PIXELS)
    ap.add_argument("--out_dir", default="eval_results")
    args = ap.parse_args()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(BASE_3B, max_pixels=args.max_pixels, min_pixels=MIN_PIXELS)
    items = pick_inputs(args.limit)
    print(f"[설정] {args.tag}  이미지 {len(items)}장  max_pixels={args.max_pixels:,}")

    # 1) 크기 — 전체 bf16
    m = load(args.checkpoint, "bfloat16")
    rec, where, nf_bf16 = run(m, processor, items, record_pos=True)
    mag = summarize(rec, where)
    del m; torch.cuda.empty_cache()

    # 2) 상실 — 언어 모델 fp16, 비전 bf16
    m = load(args.checkpoint, "float16")
    rec16, where16, nf_fp16 = run(m, processor, items, record_pos=False)
    del m; torch.cuda.empty_cache()

    worst = min((r for r in mag if r["headroom_worst"]), key=lambda r: r["headroom_worst"])
    exceed_any = max(r["exceed_rate"] for r in mag)
    print(f"[크기 bf16]  최악 여유 {worst['headroom_worst']:.2f}배 @ {worst['point']} "
          f"(max {worst['max_worst']:,.0f}, 위치 {worst['argmax_kind']})  초과 비율 최대 {exceed_any:.1%}")
    print(f"[상실 fp16]  비유한 로짓 이미지 {nf_fp16:.1%}   (bf16 기준 {nf_bf16:.1%})")

    out = {"tag": args.tag, "checkpoint": args.checkpoint, "n_images": len(items),
           "max_pixels": args.max_pixels, "magnitude_dtype": "bfloat16",
           "collapse_rate_fp16_llm": nf_fp16, "nonfinite_bf16": nf_bf16,
           "worst_point": worst["point"], "worst_headroom": worst["headroom_worst"],
           "per_point": mag,
           "fp16_per_point_exceed": {k: sum(1 for v in vs if not math.isfinite(v)) / len(vs)
                                     for k, vs in rec16.items()}}
    p = Path(args.out_dir) / f"llm_headroom_{args.tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {p}")


if __name__ == "__main__":
    main()
