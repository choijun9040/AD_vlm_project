"""
EM-VLM4AD를 NuScenes-QA val에서 평가하는 어댑터 (Exp 5-② 직접 재현 비교용)
=================================================================================
주의: EM-VLM4AD는 DriveLM만 학습했고 NuScenes-QA를 학습한 적이 없다.
      이 스크립트로 얻는 결과는 EM-VLM4AD 입장에서 **zero-shot(out-of-distribution)**
      평가이며, 우리 student(DriveLM+NuScenes-QA 학습)와 학습 데이터가 다르므로
      이 결과만으로 "이겼다/졌다"를 논문에 쓰면 안 된다. 공정 비교를 위한
      NuScenes-QA train fine-tuning은 scripts/finetune_em_vlm4ad_nuscenesqa.py 참고
      (GPU가 빌 때 실행, 지금은 스크립트만 준비).

사전 준비:
    external/EM-VLM4AD/multi_frame_results/{T5-Medium,T5-Large}/latest_model.pth
    (CHECKPOINT_PROVENANCE.md 참고 — 이미 다운로드 완료)

실행:
    python scripts/eval_em_vlm4ad_nuscenesqa.py --model-name T5-Medium --lm T5-Base --limit 50
    python scripts/eval_em_vlm4ad_nuscenesqa.py --model-name T5-Large  --lm T5-Large --limit 50
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.io import read_image
from transformers import T5Tokenizer

SCRIPTS_DIR = Path(__file__).parent
EM_VLM4AD_ROOT = SCRIPTS_DIR.parent / "external" / "EM-VLM4AD"
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(EM_VLM4AD_ROOT))

from dataloader import NuScenesQADataset, build_token_to_images  # noqa: E402
from eval_utils import normalize_answer as normalize            # noqa: E402
from modules.multi_frame_model import DriveVLMT5                # noqa: E402

DRIVELM_VAL_JSON    = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
NUSCENESQA_VAL_JSON = "data/nuscenes_qa/NuScenes_val_questions.json"

# eval.py와 정확히 동일한 전처리 (원저자 코드 그대로 재현)
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
])

# nuscenes 카메라 뷰 순서 — DriveLM image_paths dict의 원본 삽입 순서와 동일하게 맞춤
# (EM-VLM4AD는 이 순서로 학습됐으므로 순서가 바뀌면 성능이 왜곡될 수 있음)
CAMERA_ORDER = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


class Config:
    """DriveVLMT5가 요구하는 필드만 담은 최소 config (원 저장소 argparse 대체)."""

    def __init__(self, lm, gpa_hidden_size=128, lora_dim=64, lora_alpha=32, lora_dropout=0.05):
        self.lm = lm
        self.gpa_hidden_size = gpa_hidden_size
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


def load_em_vlm4ad(model_name: str, lm: str, device: str):
    config = Config(lm=lm)
    model = DriveVLMT5(config).to(device)

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base" if lm == "T5-Base" else "google-t5/t5-large")
    tokenizer.add_tokens("<")  # 원 저장소와 동일 (DriveLM 답변의 <c1,CAM_..> 마커 토큰)

    ckpt_path = EM_VLM4AD_ROOT / "multi_frame_results" / model_name / "latest_model.pth"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, image_paths_dict: dict, question: str, device: str) -> str:
    q_text = f"Question: {question} Answer:"

    img_paths = [str(image_paths_dict[cam]) for cam in CAMERA_ORDER if cam in image_paths_dict]
    imgs = torch.stack([IMG_TRANSFORM(read_image(p).float()) for p in img_paths], dim=0)
    imgs = imgs.unsqueeze(0).to(device)  # (1, 6, C, H, W)

    enc = tokenizer([q_text], padding=True, return_tensors="pt").input_ids.to(device)

    out_ids = model.generate(enc, imgs)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True,
                         help="multi_frame_results 하위 체크포인트 폴더명 "
                              "(원저자 배포: T5-Medium/T5-Large, fine-tuning 후: *_nuscenesqa_ft 등)")
    parser.add_argument("--lm", required=True, choices=["T5-Base", "T5-Large"],
                         help="DriveVLMT5 내부 백본 선택 (T5-Medium 계열=T5-Base, T5-Large 계열=T5-Large)")
    parser.add_argument("--eval_mode", default="zero_shot", choices=["zero_shot", "finetuned"],
                         help="결과 라벨용 — NuScenes-QA train으로 fine-tuning했으면 finetuned로 지정")
    parser.add_argument("--limit", type=int, default=None, help="디버깅/스모크 테스트용 샘플 수 제한")
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--out_dir", default="eval_results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[모델 로드] {args.model_name} ({args.lm} 백본, device={device})")
    model, tokenizer = load_em_vlm4ad(args.model_name, args.lm, device)

    print("[데이터 준비] DriveLM val 이미지 매핑 구성 중...")
    token_to_images = build_token_to_images(DRIVELM_VAL_JSON)

    ds = NuScenesQADataset(
        json_path=NUSCENESQA_VAL_JSON,
        token_to_images=token_to_images,
        processor=None,
    )
    samples = ds.samples
    if args.limit:
        samples = samples[: args.limit]
    mode_note = "zero-shot, EM-VLM4AD는 이 데이터로 학습된 적 없음" if args.eval_mode == "zero_shot" \
        else "NuScenes-QA train으로 fine-tuning된 체크포인트 (공정 비교)"
    print(f"평가 대상 샘플 수: {len(samples)}  ({mode_note})")

    correct_total, total = 0, 0
    correct_by_type = defaultdict(int)
    total_by_type   = defaultdict(int)
    examples = []

    for i, s in enumerate(samples):
        image_paths_dict = token_to_images[s["sample_token"]]
        pred = generate_answer(model, tokenizer, image_paths_dict, s["question"], device)
        gt = s["answer"]
        is_correct = normalize(pred) == normalize(gt)

        total += 1
        total_by_type[s["template_type"]] += 1
        if is_correct:
            correct_total += 1
            correct_by_type[s["template_type"]] += 1
        if i < 10:
            examples.append({"question": s["question"], "gt": gt, "pred": pred, "correct": is_correct})

        if (i + 1) % args.log_every == 0:
            print(f"  [{i + 1}/{len(samples)}] 누적 Accuracy: {correct_total / total * 100:.2f}%", flush=True)

    print("\n" + "=" * 50)
    print(f"모델: EM-VLM4AD ({args.model_name}, {args.lm}) — {args.eval_mode} on NuScenes-QA val")
    print(f"전체 Accuracy: {correct_total}/{total} = {correct_total / max(1, total) * 100:.2f}%")
    print("카테고리별 Accuracy:")
    for t in sorted(total_by_type):
        acc = correct_by_type[t] / total_by_type[t] * 100
        print(f"  {t:12s}: {correct_by_type[t]:5d}/{total_by_type[t]:5d} = {acc:.2f}%")
    print("=" * 50)
    print("\n예측 샘플(처음 10개):")
    for e in examples:
        print(f"  Q: {e['question']}\n    GT={e['gt']!r} PRED={e['pred']!r} match={e['correct']}")

    result = {
        "model": f"EM-VLM4AD-{args.model_name}",
        "lm": args.lm,
        "eval_mode": args.eval_mode,
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / max(1, total),
        "by_type": {
            t: {"correct": correct_by_type[t], "total": total_by_type[t], "accuracy": correct_by_type[t] / total_by_type[t]}
            for t in total_by_type
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"em_vlm4ad_{args.model_name}_{args.eval_mode}_nuscenesqa_val.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
