"""
Gemma 4 종단 확인 — 범위 확장 A5 (사전 등록: thesis_outline §6 「A의 확인 실험 A1·A5」, 2026-10-21).

`gemma-4-26B-A4B-it` 전체에서 비전 인코더만 바꿔 끼우고 같은 100문항에 답하게 한다. 언어 모델은 모두 bf16이다
(Gemma 언어 모델 자체의 fp16 문제와 섞지 않기 위해서다).

  (R) 기준       — 전체 bf16, 기준 구현 그대로(블록 뒤 연산은 fp32 승격)
  (U) 무교정     — 비전 인코더만 fp16, 블록 뒤 두 연산도 fp16(승격 없음 = 경로 (ii))
  (F) 교정       — (U)와 같되 잔차 흐름 재조정(probe_gemma4_vision.apply_residual_rescale)

    python scripts/probe_gemma4_e2e.py --offset 100 --limit 100
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from dataloader import build_token_to_images
from eval_utils import normalize_answer
from probe_gemma4_vision import apply_residual_rescale, forward_points, FP16_MAX, TARGET

REPO = "google/gemma-4-26B-A4B-it"
DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
NUSCENESQA_VAL = "data/nuscenes_qa/NuScenes_val_questions.json"
SUFFIX = "\nAnswer with a single word or number."


class NoUpcastTower(nn.Module):
    """fp16 비전 인코더를 경로 (ii)로 돌리고, 출력을 언어 쪽 dtype(bf16)으로 넘긴다."""

    def __init__(self, tower16, out_dtype):
        super().__init__()
        self.t = tower16
        self.out_dtype = out_dtype
        self.config = tower16.config

    def forward(self, pixel_values, pixel_position_ids, **_):
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        r = forward_points(self.t, pixel_values, pixel_position_ids, upcast=False)
        return BaseModelOutputWithPooling(last_hidden_state=r["final"].to(self.out_dtype))


def pick(offset, limit):
    tok = build_token_to_images(DRIVELM_VAL)
    toks = [t for t, v in tok.items() if "CAM_FRONT" in v][offset:offset + limit]
    first = {}
    for q in json.load(open(NUSCENESQA_VAL))["questions"]:
        first.setdefault(q["sample_token"], q)
    items = [(t, str(tok[t]["CAM_FRONT"]), first[t]) for t in toks if t in first]
    return items, len(toks)


@torch.no_grad()
def answer(model, processor, path, question):
    msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(path).convert("RGB")},
                                         {"type": "text", "text": question + SUFFIX}]}]
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=16, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    finite = bool(torch.isfinite(out.scores[0]).all())
    text = processor.decode(out.sequences[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, finite


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--offset", type=int, default=100)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="eval_results/gemma4_e2e_26B-A4B_off100.json")
    args = ap.parse_args()

    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    from scipy.stats import binomtest
    processor = AutoProcessor.from_pretrained(REPO)
    model = Gemma4ForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda").eval()
    items, n_img = pick(args.offset, args.limit)
    print(f"[설정] 이미지 {n_img}장 중 문항이 있는 {len(items)}개 · 해상도 기본(280)")

    orig = model.model.vision_tower
    import copy
    t16 = copy.deepcopy(orig).to(torch.float16)          # 원본(bf16)은 R 조건에 그대로 쓴다
    # 교정 c — probe_gemma4_vision과 같은 규칙(목표 여유 2.0을 넘는 가장 큰 2의 거듭제곱).
    # 크기는 A1과 같은 이미지에서 bf16(블록 뒤 fp32)으로 잰 최댓값을 쓴다.
    from probe_gemma4_vision import profile
    ip = processor.image_processor
    cache = []
    for _t, p, _q in items:
        enc = ip(images=[Image.open(p).convert("RGB")], return_tensors="pt")
        cache.append((enc["pixel_values"], enc["image_position_ids"]))
    rec, _ = profile(orig, cache, upcast=True)
    M = max(max(v for v in vals if math.isfinite(v)) for vals in rec.values())
    c = min(1.0, 2.0 ** math.floor(math.log2(FP16_MAX / (TARGET * M))))
    t16_fix = apply_residual_rescale(t16, c)
    print(f"[교정] 최대 크기 {M:,.0f} → c={c}")

    conds = {"R": orig, "U": NoUpcastTower(t16, torch.bfloat16), "F": NoUpcastTower(t16_fix, torch.bfloat16)}
    res = {k: [] for k in conds}
    for name, tower in conds.items():
        model.model.vision_tower = tower
        for tk, p, q in items:
            text, finite = answer(model, processor, p, q["question"])
            ok = normalize_answer(text) == normalize_answer(q["answer"])
            res[name].append({"token": tk, "q": q["question"], "gt": q["answer"], "pred": text,
                              "finite": finite, "invalid": (not finite) or text == "", "correct": ok})
        inv = sum(r["invalid"] for r in res[name]) / len(items)
        acc = sum(r["correct"] for r in res[name]) / len(items)
        print(f"[{name}] 무효 {inv:.1%}  정확도 {acc:.1%}")
    model.model.vision_tower = orig

    def agree(a, b):
        return sum(normalize_answer(x["pred"]) == normalize_answer(y["pred"]) for x, y in zip(res[a], res[b])) / len(items)
    b01 = sum(1 for x, y in zip(res["F"], res["R"]) if x["correct"] and not y["correct"])
    b10 = sum(1 for x, y in zip(res["F"], res["R"]) if y["correct"] and not x["correct"])
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    summ = {k: {"invalid_rate": sum(r["invalid"] for r in v) / len(items),
                "accuracy": sum(r["correct"] for r in v) / len(items)} for k, v in res.items()}
    summ["agree_F_R"] = agree("F", "R"); summ["agree_U_R"] = agree("U", "R")
    summ["mcnemar_F_vs_R"] = {"F_only": b01, "R_only": b10, "p_exact": p}
    print(f"[비교] F·R 답 일치 {summ['agree_F_R']:.1%}   McNemar(F 대 R) 불일치 {b01}/{b10}  p={p:.3f}")
    Path(args.out).write_text(json.dumps({"repo": REPO, "offset": args.offset, "n_items": len(items), "c": c,
                                          "max_magnitude": M, "summary": summ, "items": res},
                                         ensure_ascii=False, indent=1))
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
