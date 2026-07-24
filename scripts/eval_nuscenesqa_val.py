"""
NuScenes-QA val 평가 스크립트
=================================
DriveLM val(v1_1_val_nus_q_only.json)의 799개 키프레임과 매핑되는 NuScenes-QA val
샘플(약 11,309개, train과 scene 단위로 완전 분리된 held-out)에서 Student 체크포인트의
단답형 QA 정확도를 측정한다.

NuScenes-QA 정답은 전부 짧은 폐집합 문자열(yes/no, 객체명, 상태, 숫자)이므로
exact-match Accuracy를 전체 + template_type별(exist/object/status/count/comparison)로 계산한다.

실행:
    python scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/student_baseline/epoch_1
    python scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/student_full/epoch_1 --limit 200
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
from dataloader import NuScenesQADataset, build_token_to_images
from eval_utils import normalize_answer as normalize

DRIVELM_VAL_JSON    = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
NUSCENESQA_VAL_JSON = "data/nuscenes_qa/NuScenes_val_questions.json"
STUDENT_BASE        = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_model(checkpoint_path: str):
    processor = AutoProcessor.from_pretrained(
        STUDENT_BASE, max_pixels=256 * 28 * 28, min_pixels=64 * 28 * 28,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, checkpoint_path).to("cuda")
    model.eval()
    return model, processor


@torch.no_grad()
def generate_answer(model, processor, image_path: str, question: str, max_new_tokens: int) -> str:
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text",  "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="예: checkpoints/student_baseline/epoch_1")
    parser.add_argument("--limit", type=int, default=None, help="디버깅/스모크 테스트용 샘플 수 제한")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--out_dir", default="eval_results")
    args = parser.parse_args()

    print(f"[모델 로드] {args.checkpoint}")
    model, processor = load_model(args.checkpoint)

    print("[데이터 준비] DriveLM val 이미지 매핑 구성 중...")
    token_to_images = build_token_to_images(DRIVELM_VAL_JSON)

    ds = NuScenesQADataset(
        json_path=NUSCENESQA_VAL_JSON,
        token_to_images=token_to_images,
        processor=None,  # raw 모드 - generate를 직접 수행하므로 학습용 토큰화 불필요
    )
    samples = ds.samples
    if args.limit:
        samples = samples[: args.limit]
    print(f"평가 대상 샘플 수: {len(samples)}")

    correct_total, total = 0, 0
    correct_by_type = defaultdict(int)
    total_by_type   = defaultdict(int)

    for i, s in enumerate(samples):
        img_path = str(token_to_images[s["sample_token"]].get("CAM_FRONT", ""))
        pred = generate_answer(model, processor, img_path, s["question"], args.max_new_tokens)
        gt = s["answer"]
        is_correct = normalize(pred) == normalize(gt)

        total += 1
        total_by_type[s["template_type"]] += 1
        if is_correct:
            correct_total += 1
            correct_by_type[s["template_type"]] += 1

        if (i + 1) % args.log_every == 0:
            print(f"  [{i + 1}/{len(samples)}] 누적 Accuracy: {correct_total / total * 100:.2f}%", flush=True)

    print("\n" + "=" * 50)
    print(f"체크포인트: {args.checkpoint}")
    print(f"전체 Accuracy: {correct_total}/{total} = {correct_total / max(1,total) * 100:.2f}%")
    print("카테고리별 Accuracy:")
    for t in sorted(total_by_type):
        acc = correct_by_type[t] / total_by_type[t] * 100
        print(f"  {t:12s}: {correct_by_type[t]:5d}/{total_by_type[t]:5d} = {acc:.2f}%")
    print("=" * 50)

    result = {
        "checkpoint": args.checkpoint,
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / max(1, total),
        "by_type": {
            t: {
                "correct": correct_by_type[t],
                "total": total_by_type[t],
                "accuracy": correct_by_type[t] / total_by_type[t],
            }
            for t in total_by_type
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ckpt_tag = Path(args.checkpoint).parent.name + "_" + Path(args.checkpoint).name
    out_path = out_dir / f"{ckpt_tag}_nuscenesqa_val.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
