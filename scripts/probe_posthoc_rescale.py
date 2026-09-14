"""
사후 리스케일 실험 — 여유를 학습 없이 얻을 수 있는가
=======================================================
Qwen2.5-VL의 patch merger는 앞에 RMSNorm(`ln_q`)이 있어 blk31 출력이 토큰별로 정규화된 뒤
LLM에 전달된다. **균일 스케일은 정규화가 흡수하므로** blk31의 MLP 출력을 일정 배수로 나눠도
표현이 수학적으로 거의 동일할 수 있다. 사실이면 fp16 여유를 **학습 없이** 얻는다는 뜻이다.

해석 방침은 `thesis_outline_20260910.md` §6에 결과 확인 전 사전 등록되어 있다.

두 가지를 잰다.
  (i) 원본 해상도에서 fp16 붕괴율 — 리스케일이 붕괴를 막는가
  (ii) NuScenes-QA 정확도 — bf16 기준으로, 리스케일이 기능을 해치는가

실행:
    python scripts/probe_posthoc_rescale.py --scale 3 --n_images 250 --n_qa 300
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images, NuScenesQADataset
from profile_vision_activations import (
    DRIVELM_VAL_JSON, BASE_3B, MAX_PIXELS, MIN_PIXELS,
    build_pixel_cache, get_visual, percentile, FP16_MAX,
)
from eval_utils import normalize_answer as normalize

CKPT = "checkpoints/student_baseline_v2/epoch_1"
NUSCENESQA_VAL = "data/nuscenes_qa/NuScenes_val_questions.json"


def attach(model, scale, mode="out"):
    """blk31 MLP 개입.

    mode="out"  : MLP **출력**을 1/scale (= fixB). 이미 실패한 조건.
    mode="gateup": gate·up 출력을 각각 1/sqrt(scale) → 곱이 1/scale (= fixA).
                   MLP에 bias가 없어 gate/up **가중치**를 1/sqrt(scale)로 나누는 것과 등가다.
    """
    if scale == 1.0:
        return []
    mlp = get_visual(model).blocks[31].mlp

    def scaler(f):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            sc = t * f
            return sc if not isinstance(out, tuple) else (sc,) + out[1:]
        return fn

    if mode == "gateup":
        f = 1.0 / math.sqrt(scale)
        return [mlp.gate_proj.register_forward_hook(scaler(f)),
                mlp.up_proj.register_forward_hook(scaler(f))]
    return [mlp.register_forward_hook(scaler(1.0 / scale))]


@torch.no_grad()
def collapse_rate(model, cache, scale, mode="out"):
    """fp16으로 통과시켜 이미지 단위 NaN 발생률과 blk31 크기를 잰다."""
    visual = get_visual(model.to(torch.float16))
    hs = attach(model, scale, mode)
    rec = {}
    h31 = visual.blocks[31].register_forward_hook(
        lambda m, i, o: rec.__setitem__("b31", (o[0] if isinstance(o, tuple) else o).detach()))

    nan, mags = 0, []
    for pv, grid in cache:
        rec.clear()
        out = visual(pv.to("cuda", torch.float16), grid.to("cuda"))
        b = rec["b31"].float()
        if torch.isnan(b).any() or torch.isinf(b).any():
            nan += 1
        else:
            mags.append(b.abs().max().item())
        if torch.isnan(out.float()).any():
            pass
    h31.remove()
    for x in hs:
        x.remove()
    model.to(torch.bfloat16)
    return {"nan_rate": nan / len(cache), "nan": nan, "n": len(cache),
            "blk31_p50": percentile(sorted(mags), 0.5) if mags else None}


@torch.no_grad()
def qa_accuracy(model, processor, samples, token_to_images, scale, mode="out",
                max_new_tokens=16):
    """bf16에서 정확도 — 리스케일이 기능을 해치는지 본다."""
    hs = attach(model, scale, mode)
    dev = next(model.parameters()).device
    correct = 0
    for s in samples:
        img = str(token_to_images[s["sample_token"]]["CAM_FRONT"])
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": s["question"]}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = processor(text=[text], images=[Image.open(img).convert("RGB")],
                        return_tensors="pt").to(dev)
        ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id)
        pred = processor.batch_decode(ids[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()
        correct += int(normalize(pred) == normalize(s["answer"]))
    for x in hs:
        x.remove()
    return correct / len(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=3.0, help="blk31 MLP 출력을 1/scale로")
    ap.add_argument("--n_images", type=int, default=250)
    ap.add_argument("--n_qa", type=int, default=300)
    ap.add_argument("--conds", default="none,fixb,fixa",
                    help="측정할 조건 (none/fixb/fixa 쉼표 구분)")
    ap.add_argument("--out", default="eval_results/posthoc_rescale.json")
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.n_images]

    print("[모델 로드]", flush=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE_3B, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, CKPT).to("cuda").eval()

    # --- (i) 붕괴율: 원본 해상도 ---
    proc_native = AutoProcessor.from_pretrained(BASE_3B, max_pixels=1440000, min_pixels=3136)
    cache = build_pixel_cache(proc_native, paths, 16)

    res = {"checkpoint": CKPT, "scale": args.scale, "collapse": {}, "accuracy": {}}
    ALL_CONDS = [("무개입", 1.0, "out"),
                 (f"fixB 출력 1/{args.scale:g}", args.scale, "out"),
                 (f"fixA 곱 1/{args.scale:g}", args.scale, "gateup")]
    # fixB는 no-op임이 이미 확인돼 전체 평가에서는 뺄 수 있다
    keep = {"none": "무개입", "fixb": "fixB", "fixa": "fixA"}
    want = [keep[c.strip().lower()] for c in args.conds.split(",")]
    CONDS = [c for c in ALL_CONDS if any(c[0].startswith(w) for w in want)]
    for label, sc, md in CONDS:
        r = collapse_rate(model, cache, sc, md)
        res["collapse"][label] = r
        print(f"  [{label:16s}] fp16 붕괴 {r['nan']}/{r['n']} = {r['nan_rate']*100:5.1f}%   "
              f"blk31 p50={r['blk31_p50'] or float('nan'):,.0f}", flush=True)
    del cache

    # --- (ii) 정확도: 평가 해상도, bf16 ---
    proc_eval = AutoProcessor.from_pretrained(BASE_3B, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS)
    ds = NuScenesQADataset(json_path=NUSCENESQA_VAL, token_to_images=tok, processor=None)
    samples = ds.samples[: args.n_qa]
    print(f"\n[정확도] NuScenes-QA val {len(samples)}문항 (bf16)", flush=True)
    for label, sc, md in CONDS:
        a = qa_accuracy(model, proc_eval, samples, tok, sc, md)
        res["accuracy"][label] = a
        print(f"  [{label:16s}] {a*100:.2f}%", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    base_acc = res["accuracy"]["무개입"]
    print("\n" + "=" * 74)
    print(f"{'조건':22s}{'붕괴율':>10}{'정확도':>10}{'Δ정확도':>10}")
    print("-" * 74)
    for label, _, _ in CONDS:
        c = res["collapse"][label]["nan_rate"] * 100
        a = res["accuracy"][label]
        print(f"{label:22s}{c:>9.1f}%{a*100:>9.2f}%{(a-base_acc)*100:>+9.2f}pp")
    print("사전 등록된 해석: thesis_outline_20260910.md §6 '사후 리스케일 실험'")
    print("=" * 74)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
