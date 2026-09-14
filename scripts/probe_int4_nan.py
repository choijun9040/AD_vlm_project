"""
INT4 수치 붕괴 지점 추적 프로브
=================================
AWQ INT4 체크포인트가 특정 이미지에서만 `!!!!`(토큰 id 0) 만 뱉는 현상의 원인을
찾는다. 생성이 아니라 **prefill 한 번**만 돌리면서 vision tower와 LLM 디코더의
매 층 출력에서 NaN/Inf 발생 여부와 최대 절댓값을 기록해, 수치가 처음 터지는
지점을 특정한다.

같은 이미지를 FP16(LoRA) 체크포인트로도 돌려 대조한다.

실행:
    python scripts/probe_int4_nan.py \
        --int4 checkpoints_awq/student_baseline_v2 \
        --fp16 checkpoints/student_baseline_v2/epoch_1 \
        --bad-token f56a544064a548a39a81f18cc8f633c5 \
        --good-token 6b9af38cbf654171870cb14b4ae89fd6
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from eval_nuscenesqa_val import (
    DRIVELM_VAL_JSON, load_model, detect_precision,
)
from dataloader import build_token_to_images

QUESTION = "Are there any traffic cones?"


def stats(t: torch.Tensor) -> str:
    if not torch.is_tensor(t):
        return "-"
    f = t.float()
    return (f"max|x|={f.abs().max().item():>10.2f}  "
            f"nan={int(torch.isnan(f).sum()):>6d}  inf={int(torch.isinf(f).sum()):>6d}")


@torch.no_grad()
def probe(model, processor, image_path: str, label: str):
    device = next(model.parameters()).device
    records = []

    def hook(name):
        def fn(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                records.append((name, t.detach()))
        return fn

    core = model.model if hasattr(model, "model") else model
    core = getattr(core, "model", core)          # PeftModel → base → Qwen2_5_VLModel
    visual = core.visual if hasattr(core, "visual") else model.visual
    layers = core.language_model.layers if hasattr(core, "language_model") else core.layers

    handles = [visual.blocks[i].register_forward_hook(hook(f"visual.blocks[{i}]"))
               for i in range(0, len(visual.blocks), 8)]
    handles.append(visual.register_forward_hook(hook("visual(out)")))
    handles += [layers[i].register_forward_hook(hook(f"decoder[{i}]"))
                for i in range(0, len(layers), 4)]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": QUESTION},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[Image.open(image_path).convert("RGB")],
                       return_tensors="pt").to(device)
    out = model(**inputs)

    for h in handles:
        h.remove()

    print(f"\n----- {label} -----")
    print(f"  vision tokens: {int((inputs['input_ids'] == 151655).sum())}")
    for name, t in records:
        print(f"  {name:<22s} {stats(t)}")
    logits = out.logits[0, -1]
    print(f"  {'logits(last)':<22s} {stats(logits)}")
    print(f"  argmax={int(logits.float().argmax())}  "
          f"top5={[int(i) for i in logits.float().topk(5).indices]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int4", default="checkpoints_awq/student_baseline_v2")
    ap.add_argument("--fp16", default="checkpoints/student_baseline_v2/epoch_1")
    ap.add_argument("--bad-token", dest="bad_token", required=True)
    ap.add_argument("--good-token", dest="good_token", required=True)
    args = ap.parse_args()

    token_to_images = build_token_to_images(DRIVELM_VAL_JSON)
    paths = {
        "붕괴 이미지": str(token_to_images[args.bad_token]["CAM_FRONT"]),
        "정상 이미지": str(token_to_images[args.good_token]["CAM_FRONT"]),
    }

    for ckpt in [args.int4, args.fp16]:
        prec = detect_precision(ckpt)
        print(f"\n{'=' * 70}\n[{ckpt}]  precision={prec}\n{'=' * 70}")
        model, processor = load_model(ckpt, prec)
        for label, p in paths.items():
            probe(model, processor, p, f"{label} ({Path(p).name})")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
