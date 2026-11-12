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
import awq_compat  # 2026-10-22: autoawq가 설치되면 peft가 awq를 import하므로 qwen3·datasets 스텁이 필요하다

BASE_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
NUSCENESQA_VAL = "data/nuscenes_qa/NuScenes_val_questions.json"
FP16_MAX = 65504.0
NATIVE_MAX_PIXELS = 1_440_000          # 원본 해상도 (headroom_guard 표 30·E1과 같은 값)
MIN_PIXELS = 64 * 28 * 28              # 평가 스크립트와 같은 하한
IMAGE_TOKEN_ID = 151655


ATTN = {"impl": "eager"}          # 개정 24: --attn default면 지정 안 함(공식 사용법, sdpa)


def load(checkpoint, dtype):
    awq_compat.patch()
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel
    kw = {} if ATTN["impl"] == "default" else {"attn_implementation": ATTN["impl"]}
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_3B, torch_dtype=getattr(torch, dtype), **kw)
    if checkpoint and checkpoint.lower() != "none":
        m = PeftModel.from_pretrained(m, checkpoint).merge_and_unload()
    m = m.eval().to("cuda")
    # 비전 인코더는 언제나 bf16 — 백본의 상실만 보기 위해서다.
    m.visual.to(torch.bfloat16)
    return m


def pick_inputs(limit, offset=0):
    """headroom_guard와 같은 이미지 순서(DriveLM 검증 키프레임 순), 이미지마다 첫 NuScenes-QA 문항.

    offset: 그 순서에서 시작 위치 — [offset:offset+limit]. (검증 키프레임 799장 모두 문항이 있어
    문항 필터가 순서를 바꾸지 않는다.)"""
    tok = build_token_to_images(DRIVELM_VAL)
    qs = json.load(open(NUSCENESQA_VAL))["questions"]
    first_q = {}
    for q in qs:
        first_q.setdefault(q["sample_token"], q["question"])
    items = []
    for t, cams in tok.items():
        if "CAM_FRONT" in cams and t in first_q:
            items.append((t, str(cams["CAM_FRONT"]), first_q[t]))
        if len(items) >= offset + limit:
            break
    return items[offset:]


def all_items():
    """정확도 평가와 같은 11,309문항(`eval_nuscenesqa_val.py`의 NuScenesQADataset 순서) — 개정 13 (2026-11-04)."""
    from dataloader import NuScenesQADataset
    tok = build_token_to_images(DRIVELM_VAL)
    ds = NuScenesQADataset(json_path=NUSCENESQA_VAL, token_to_images=tok, processor=None)
    return [(s["sample_token"], str(tok[s["sample_token"]]["CAM_FRONT"]), s["question"]) for s in ds.samples]


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
def run(model, processor, items, record_pos, logit_flags=None, cache_images=False):
    """logit_flags: 리스트를 주면 이미지별 비유한 로짓 여부를 덧붙인다 — 집계에는 영향 없음.

    cache_images (개정 13, 2026-11-04): 11,309문항은 799장을 나눠 쓴다. 이미지 전처리와 비전 인코더 출력을 이미지마다 한 번만 계산해
    그 이미지의 문항에 다시 쓴다 — 처리기가 하는 이미지 토큰 펼치기를 그대로 하고(같은 input_ids), 비전 출력은 forward 안의
    masked_scatter와 같은 자리에 넣는다. RoPE 위치는 past_key_values가 없어 문항마다 input_ids·image_grid_thw로 다시 계산된다."""
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
    img_cache = {}
    merge = processor.image_processor.merge_size ** 2
    for n_done, (_tok, path, q) in enumerate(items):
        if n_done and n_done % 500 == 0:
            print(f"  {n_done}/{len(items)}", flush=True)
        msgs = [{"role": "user", "content": [{"type": "image", "image": path},
                                             {"type": "text", "text": q}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if not cache_images:
            inp = processor(text=[text], images=[Image.open(path).convert("RGB")],
                            return_tensors="pt").to("cuda")
            cur["ids"] = inp["input_ids"][0]
            out = model(**inp)
        else:
            if path not in img_cache:
                enc = processor.image_processor(images=[Image.open(path).convert("RGB")], return_tensors="pt")
                grid = enc["image_grid_thw"].to("cuda")
                emb = model.visual(enc["pixel_values"].to("cuda", model.visual.dtype), grid_thw=grid)
                img_cache[path] = (grid.cpu(), emb.cpu())
            grid, emb = img_cache[path]
            n_tok = int(grid.prod()) // merge
            t2 = text.replace("<|image_pad|>", "<|image_pad|>" * n_tok, 1)
            tk = processor.tokenizer([t2], return_tensors="pt").to("cuda")
            ids = tk["input_ids"]
            emb_in = model.model.embed_tokens(ids)
            msk = (ids == IMAGE_TOKEN_ID).unsqueeze(-1).expand_as(emb_in)
            emb_in = emb_in.masked_scatter(msk, emb.to("cuda", emb_in.dtype))
            cur["ids"] = ids[0]
            out = model(input_ids=ids, inputs_embeds=emb_in, attention_mask=tk["attention_mask"],
                        image_grid_thw=grid.to("cuda"))
        nfl = not torch.isfinite(out.logits).all()
        if nfl:
            nonfinite_logits += 1
        if logit_flags is not None:
            logit_flags.append(bool(nfl))
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
    ap.add_argument("--offset", type=int, default=0, help="이미지 시작 위치 (표준 순서 기준)")
    ap.add_argument("--all_questions", action="store_true",
                    help="이미지마다 첫 문항 대신 정확도 평가와 같은 11,309문항 전부(--limit·--offset 무시) — 개정 13")
    ap.add_argument("--cache_images", action="store_true",
                    help="이미지 전처리·비전 인코더 출력을 이미지마다 한 번만 계산(개정 13 — 결과는 같고 반복만 없앤다)")
    ap.add_argument("--first_n", type=int, default=0, help="문항 앞 N개만(동일성 시험용)")
    ap.add_argument("--min_pixels", type=int, default=MIN_PIXELS, help="공식 처리기 기본값은 3,136 — 개정 24")
    ap.add_argument("--rwqa", default=None, help="RealWorldQA 문항 jsonl — 765문항 전부(문항당 이미지 하나). 개정 24, 2026-11-04")
    ap.add_argument("--attn", default="eager", help="어텐션 구현. default = 지정 안 함(공식 사용법, sdpa) — 개정 24")
    ap.add_argument("--per_image", default=None,
                    help="이미지별 지점 max(bf16·fp16, 비유한이면 inf)·bf16 argmax 위치 종류·비유한 로짓 여부와 "
                         "장면 토큰을 jsonl로 저장('w'). 장면 군집 부트스트랩용")
    args = ap.parse_args()

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(BASE_3B, max_pixels=args.max_pixels, min_pixels=args.min_pixels)
    ATTN["impl"] = args.attn
    if args.rwqa:
        items = [(str(r["idx"]), r["image"], r["question"]) for r in map(json.loads, open(args.rwqa))]
    else:
        items = all_items() if args.all_questions else pick_inputs(args.limit, args.offset)
    if args.first_n:
        items = items[:args.first_n]
    print(f"[설정] {args.tag}  {'문항' if args.all_questions else '이미지'} {len(items)}개  max_pixels={args.max_pixels:,}")

    # 1) 크기 — 전체 bf16
    m = load(args.checkpoint, "bfloat16")
    flags_bf16 = [] if args.per_image else None
    rec, where, nf_bf16 = run(m, processor, items, record_pos=True, logit_flags=flags_bf16, cache_images=args.cache_images)
    mag = summarize(rec, where)
    del m; torch.cuda.empty_cache()

    # 2) 상실 — 언어 모델 fp16, 비전 bf16
    m = load(args.checkpoint, "float16")
    flags_fp16 = [] if args.per_image else None
    rec16, where16, nf_fp16 = run(m, processor, items, record_pos=False, logit_flags=flags_fp16, cache_images=args.cache_images)
    del m; torch.cuda.empty_cache()

    worst = min((r for r in mag if r["headroom_worst"]), key=lambda r: r["headroom_worst"])
    exceed_any = max(r["exceed_rate"] for r in mag)
    print(f"[크기 bf16]  최악 여유 {worst['headroom_worst']:.2f}배 @ {worst['point']} "
          f"(max {worst['max_worst']:,.0f}, 위치 {worst['argmax_kind']})  초과 비율 최대 {exceed_any:.1%}")
    print(f"[상실 fp16]  비유한 로짓 이미지 {nf_fp16:.1%}   (bf16 기준 {nf_bf16:.1%})")

    out = {"tag": args.tag, "checkpoint": args.checkpoint, "n_images": len({it[1] for it in items}), "n_items": len(items),
           "all_questions": args.all_questions, "cache_images": args.cache_images,
           "max_pixels": args.max_pixels, "magnitude_dtype": "bfloat16", "rwqa": args.rwqa, "attn": args.attn,
           "collapse_rate_fp16_llm": nf_fp16, "nonfinite_bf16": nf_bf16,
           "worst_point": worst["point"], "worst_headroom": worst["headroom_worst"],
           "per_point": mag,
           "fp16_per_point_exceed": {k: sum(1 for v in vs if not math.isfinite(v)) / len(vs)
                                     for k, vs in rec16.items()}}
    p = Path(args.out_dir) / f"llm_headroom_{args.tag}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"→ {p}")
    if args.per_image:
        # fp16 통과는 위치를 기록하지 않으므로(record_pos=False) argmax 종류는 bf16만 남긴다
        from scene_map import image_to_scene
        sc = image_to_scene()
        with open(args.per_image, "w") as fo:
            for j, (tok_, path, q) in enumerate(items):
                nm = Path(path).name
                fo.write(json.dumps({
                    "tag": args.tag, "checkpoint": args.checkpoint, "max_pixels": args.max_pixels,
                    "image": nm, "scene": sc.get(nm), "sample_token": tok_, "question": q,
                    "point_max_bf16": {k: v[j] for k, v in rec.items()},
                    "argmax_kind_bf16": {k: v[j] for k, v in where.items()},
                    "point_max_fp16_llm": {k: v[j] for k, v in rec16.items()},
                    "nonfinite_logits_bf16": flags_bf16[j], "nonfinite_logits_fp16_llm": flags_fp16[j],
                }) + "\n")
        print(f"이미지별: {args.per_image}")


if __name__ == "__main__":
    main()
