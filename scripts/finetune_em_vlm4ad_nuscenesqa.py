"""
EM-VLM4AD를 NuScenes-QA train으로 fine-tuning — Exp 5-② 공정 비교 완성용
==================================================================================
EM-VLM4AD는 DriveLM만 학습했고 우리 student는 DriveLM+NuScenes-QA를 학습했으므로,
zero-shot 비교(eval_em_vlm4ad_nuscenesqa.py --eval_mode zero_shot)는 학습 데이터가
달라 공정하지 않다. 기존 DriveLM 체크포인트에서 이어서 NuScenes-QA train으로 가볍게
fine-tuning하면 "동일 학습 데이터, 동일 평가셋" 비교가 완성된다.

주의: 이 스크립트는 GPU가 필요하다. 지금은 골격만 준비해두고, Full 학습(또는 이후
+spatial/+temporal 학습)이 GPU를 비우는 시점에 실행할 것.

실행 (GPU 필요):
    python scripts/finetune_em_vlm4ad_nuscenesqa.py --model-name T5-Medium --lm T5-Base \
        --epochs 3 --batch-size 8

완료 후 평가:
    python scripts/eval_em_vlm4ad_nuscenesqa.py --model-name T5-Medium_nuscenesqa_ft \
        --lm T5-Base --eval_mode finetuned
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from transformers import T5Tokenizer

SCRIPTS_DIR = Path(__file__).parent
EM_VLM4AD_ROOT = SCRIPTS_DIR.parent / "external" / "EM-VLM4AD"
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(EM_VLM4AD_ROOT))

from dataloader import build_token_to_images  # noqa: E402
from modules.multi_frame_model import DriveVLMT5  # noqa: E402

NUSCENESQA_TRAIN_JSON = "data/nuscenes_qa/NuScenes_train_questions.json"
DRIVELM_TRAIN_JSON    = "data/QA_dataset_nus/v1_0_train_nus.json"  # 우리 student 학습에 쓴 것과 동일 이미지 소스

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
])
CAMERA_ORDER = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


class NuScenesQAForEMVLM4AD(Dataset):
    """NuScenes-QA train을 EM-VLM4AD 입력 포맷(6뷰 이미지 + Q/A 텍스트)으로 변환.

    train_distillation.py의 NuScenesQADataset과 동일한 필터링 원칙(token_to_images에
    존재하는 sample_token만 사용)을 따르므로, 우리 student가 학습 때 본 것과 같은
    NuScenes-QA train 서브셋(54,607개)을 그대로 사용하게 된다.
    """

    def __init__(self, json_path, token_to_images, tokenizer):
        with open(json_path) as f:
            data = json.load(f)
        self.samples = [q for q in data["questions"] if q["sample_token"] in token_to_images]
        self.token_to_images = token_to_images
        self.tokenizer = tokenizer
        print(f"[NuScenesQAForEMVLM4AD] 매핑 가능: {len(self.samples)}/{len(data['questions'])}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image_paths_dict = self.token_to_images[s["sample_token"]]
        img_paths = [str(image_paths_dict[cam]) for cam in CAMERA_ORDER if cam in image_paths_dict]
        imgs = torch.stack([IMG_TRANSFORM(read_image(p).float()) for p in img_paths], dim=0)
        q_text = f"Question: {s['question']} Answer:"
        return q_text, imgs, s["answer"]

    def collate_fn(self, batch):
        q_texts, imgs, a_texts = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        encodings = self.tokenizer(list(q_texts), padding=True, return_tensors="pt").input_ids
        labels = self.tokenizer(list(a_texts), padding=True, return_tensors="pt").input_ids
        return encodings, imgs, labels


class Config:
    """DriveVLMT5가 요구하는 필드만 담은 최소 config (원 저장소 argparse 대체)."""

    def __init__(self, lm, gpa_hidden_size=128, lora_dim=64, lora_alpha=32, lora_dropout=0.05):
        self.lm = lm
        self.gpa_hidden_size = gpa_hidden_size
        self.lora_dim = lora_dim
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True,
                         help="이어서 학습할 원본 체크포인트 폴더명 (T5-Medium 또는 T5-Large)")
    parser.add_argument("--lm", required=True, choices=["T5-Base", "T5-Large"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--out-name", default=None, help="저장 폴더명 (기본: {model-name}_nuscenesqa_ft)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_name = args.out_name or f"{args.model_name}_nuscenesqa_ft"
    out_dir = EM_VLM4AD_ROOT / "multi_frame_results" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[모델 로드] {args.model_name}({args.lm}) 체크포인트에서 이어서 시작, device={device}")
    config = Config(lm=args.lm)
    model = DriveVLMT5(config).to(device)
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base" if args.lm == "T5-Base" else "google-t5/t5-large")
    tokenizer.add_tokens("<")
    ckpt_path = EM_VLM4AD_ROOT / "multi_frame_results" / args.model_name / "latest_model.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("[데이터 준비] NuScenes-QA train (DriveLM train 이미지와 매핑, 우리 student와 동일 서브셋)")
    token_to_images = build_token_to_images(DRIVELM_TRAIN_JSON)
    train_ds = NuScenesQAForEMVLM4AD(NUSCENESQA_TRAIN_JSON, token_to_images, tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_ds.collate_fn, num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"\n{'=' * 50}")
    print(f"NuScenes-QA fine-tuning 시작: {out_name}")
    print(f"  epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.learning_rate}")
    print(f"  steps/epoch: {len(train_loader)}")
    print(f"{'=' * 50}\n")

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        t0 = time.time()
        for step, (enc, imgs, labels) in enumerate(train_loader):
            enc, imgs, labels = enc.to(device), imgs.to(device), labels.to(device)

            outputs = model(enc, imgs, labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

            if (step + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  [epoch {epoch}] step {step + 1}/{len(train_loader)} "
                    f"avg_loss={epoch_loss / (step + 1):.4f} ({elapsed:.0f}s 경과)",
                    flush=True,
                )

        print(f"[Epoch {epoch} 완료] avg_loss={epoch_loss / len(train_loader):.4f}")
        torch.save(model.state_dict(), out_dir / "latest_model.pth")
        print(f"  체크포인트 저장: {out_dir / 'latest_model.pth'}")

    print("\nfine-tuning 완료. 평가는 다음으로 실행:")
    print(
        f"  python scripts/eval_em_vlm4ad_nuscenesqa.py --model-name {out_name} "
        f"--lm {args.lm} --eval_mode finetuned"
    )


if __name__ == "__main__":
    main()
