"""
Teacher(7B) count 카테고리 진단 스크립트
=================================
5-way student 결과에서 count 카테고리가 5개 전부 7~11%대로 공통 약점이었음 —
"모델 스케일 문제인지 카운팅 전용 supervision이 필요한 문제인지" 판단하기 위해
teacher(7B, LoRA FT) 자체의 count 정확도를 먼저 확인한다. teacher가 이미 7B로
student(3B)보다 크므로, teacher도 낮으면 스케일로 해결될 문제가 아니라는 뜻이 된다.

NuScenes-QA val 전체가 아니라 template_type == "count"인 샘플만 평가한다
(2,212개, 5-way student eval과 동일한 정규화/채점 로직 재사용).

실행:
    python scripts/eval_teacher_count.py
"""

import json
import sys
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
TEACHER_BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
TEACHER_LORA = "checkpoints/teacher_lora/epoch_1"


def load_teacher():
    processor = AutoProcessor.from_pretrained(
        TEACHER_BASE, max_pixels=256 * 28 * 28, min_pixels=64 * 28 * 28,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        TEACHER_BASE, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, TEACHER_LORA)
    model.eval()
    return model, processor


@torch.no_grad()
def generate_answer(model, processor, image_path: str, question: str, max_new_tokens: int = 16) -> str:
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    out_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def main():
    print(f"[모델 로드] teacher: {TEACHER_BASE} + {TEACHER_LORA}")
    model, processor = load_teacher()

    print("[데이터 준비] DriveLM val 이미지 매핑 구성 중...")
    token_to_images = build_token_to_images(DRIVELM_VAL_JSON)
    ds = NuScenesQADataset(json_path=NUSCENESQA_VAL_JSON, token_to_images=token_to_images, processor=None)

    count_samples = [s for s in ds.samples if s["template_type"] == "count"]
    print(f"count 카테고리 샘플 수: {len(count_samples)}")

    correct, total = 0, 0
    examples = []
    for i, s in enumerate(count_samples):
        img_path = str(token_to_images[s["sample_token"]].get("CAM_FRONT", ""))
        pred = generate_answer(model, processor, img_path, s["question"])
        gt = s["answer"]
        is_correct = normalize(pred) == normalize(gt)
        total += 1
        if is_correct:
            correct += 1
        if i < 15:
            examples.append({"question": s["question"], "gt": gt, "pred": pred, "correct": is_correct})
        if (i + 1) % 200 == 0:
            print(f"  [{i + 1}/{len(count_samples)}] 누적 Accuracy: {correct / total * 100:.2f}%", flush=True)

    print("\n" + "=" * 50)
    print(f"Teacher(7B) count Accuracy: {correct}/{total} = {correct / max(1,total) * 100:.2f}%")
    print("=" * 50)
    print("\n예측 샘플(처음 15개):")
    for e in examples:
        print(f"  Q: {e['question']}\n    GT={e['gt']!r} PRED={e['pred']!r} match={e['correct']}")

    result = {
        "model": "teacher_7B_LoRA",
        "category": "count",
        "total": total,
        "correct": correct,
        "accuracy": correct / max(1, total),
    }
    out_path = Path("eval_results/teacher_count_diagnosis.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
