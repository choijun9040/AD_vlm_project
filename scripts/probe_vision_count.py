"""
Option C 사전 검증 — frozen vision feature에 개수 정보가 선형적으로 존재하는가
==============================================================================
Option C(density map / attention pooling 카운팅 head)를 정식 구현하기 전에,
"vision encoder patch feature 자체에 개수 정보가 있는지"만 훨씬 싸게 확인한다.

방법: student(Qwen2.5-VL-3B)를 완전히 freeze한 채(그래디언트 없음, LoRA도 없음)
vision encoder 중간층(layer 23 — L_align과 동일 레이어)의 patch feature를
평균 pooling해서 하나의 벡터로 만들고, 그 위에 아주 작은 linear regression만
학습시켜 log(1+count)를 예측한다. backbone을 전혀 학습시키지 않으므로 feature
추출은 1회만 하면 되고(캐싱), probe 학습 자체는 수 초 내로 끝난다.

판단 기준: held-out MSE가 "항상 평균값만 예측하는" baseline보다 확실히 낮고,
예측-정답 상관계수(corr)가 0에서 유의미하게 벗어나면 "vision feature에 신호
있음" → Option C 정식 구현 근거 확보. 그렇지 않으면 Option C도 밑빠진 독일
가능성이 높다는 뜻.

실행:
    python scripts/probe_vision_count.py
"""

import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

sys.path.append(str(Path(__file__).parent))
from dataloader import NuScenesQADataset, build_token_to_images

STUDENT_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
SPATIAL_LAYER_IDX = 23  # L_align과 동일 레이어 — 이미 검증된 위치
VISION_HIDDEN = 1280
DRIVELM_TRAIN_JSON = "data/QA_dataset_nus/v1_0_train_nus.json"
NUSCENESQA_TRAIN_JSON = "data/nuscenes_qa/NuScenes_train_questions.json"

N_SAMPLES = 2000
HELDOUT_N = 300
SEED = 42

captured = {}


def hook(module, inp, output):
    captured["feat"] = output.detach()


def parse_count(answer_str):
    m = re.match(r"^-?\d+$", answer_str.strip())
    return float(m.group()) if m else None


def main():
    device = "cuda"
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[모델 로드] {STUDENT_BASE} (frozen, LoRA 없음)")
    processor = AutoProcessor.from_pretrained(
        STUDENT_BASE, max_pixels=256 * 28 * 28, min_pixels=64 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    h = model.visual.blocks[SPATIAL_LAYER_IDX].register_forward_hook(hook)

    print("[데이터 준비] NuScenes-QA train에서 count 타입 샘플링")
    token_to_images = build_token_to_images(DRIVELM_TRAIN_JSON)
    ds = NuScenesQADataset(json_path=NUSCENESQA_TRAIN_JSON, token_to_images=token_to_images, processor=None)
    count_samples = [s for s in ds.samples if s["template_type"] == "count"]
    count_samples = [s for s in count_samples if parse_count(s["answer"]) is not None]
    random.shuffle(count_samples)
    pool = count_samples[:N_SAMPLES]
    print(f"feature 추출 대상: {len(pool)}개")

    feats, counts = [], []
    with torch.no_grad():
        for i, s in enumerate(pool):
            img_path = str(token_to_images[s["sample_token"]].get("CAM_FRONT", ""))
            image = Image.open(img_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": s["question"]},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

            _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                image_grid_thw=inputs["image_grid_thw"],
            )
            patch_feat = captured["feat"]
            if i == 0:
                print(f"  [디버그] hook 캡처 shape: {tuple(patch_feat.shape)}")
            # (num_patches, hidden) 또는 (1, num_patches, hidden) 모두 대응
            if patch_feat.dim() == 3:
                patch_feat = patch_feat.squeeze(0)
            pooled = patch_feat.mean(dim=0)  # (hidden,) — 평균 pooling

            feats.append(pooled.float().cpu())
            counts.append(parse_count(s["answer"]))

            if (i + 1) % 200 == 0:
                print(f"  추출 {i + 1}/{len(pool)}", flush=True)

    h.remove()

    X = torch.stack(feats)                          # (N, 1280)
    counts_t = torch.tensor(counts, dtype=torch.float32)
    y = torch.log1p(counts_t)                        # log(1+count) 회귀 (옵션 A에서 검증된 스케일링)

    N = X.size(0)
    perm = torch.randperm(N)
    heldout_idx, train_idx = perm[:HELDOUT_N], perm[HELDOUT_N:]

    Xtr, ytr = X[train_idx], y[train_idx]
    Xho, yho = X[heldout_idx], y[heldout_idx]
    counts_ho = counts_t[heldout_idx]

    mu, std = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr_n, Xho_n = (Xtr - mu) / std, (Xho - mu) / std

    probe = nn.Linear(VISION_HIDDEN, 1)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)

    print(f"\n{'=' * 60}\nlinear probe 학습 (train={len(train_idx)}, held-out={len(heldout_idx)})\n{'=' * 60}\n")

    for epoch in range(300):
        opt.zero_grad()
        pred = probe(Xtr_n).squeeze(-1)
        loss = F.mse_loss(pred, ytr)
        loss.backward()
        opt.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred_ho = probe(Xho_n).squeeze(-1)
                loss_ho = F.mse_loss(pred_ho, yho).item()
                pred_count = torch.round(torch.expm1(pred_ho)).clamp(min=0)
                acc = (pred_count == counts_ho).float().mean().item() * 100
                corr = torch.corrcoef(torch.stack([pred_ho, yho]))[0, 1].item()
            print(
                f"  epoch {epoch + 1}/300 | train_loss={loss.item():.4f} heldout_loss={loss_ho:.4f} "
                f"heldout_round_acc={acc:.1f}% corr(pred,gt)={corr:.3f}",
                flush=True,
            )

    # 베이스라인: held-out에 대해 그냥 train 평균 count만 예측했을 때
    mean_pred = ytr.mean()
    baseline_loss = F.mse_loss(mean_pred.expand_as(yho), yho).item()
    baseline_count_pred = torch.round(torch.expm1(mean_pred)).clamp(min=0)
    baseline_acc = (baseline_count_pred == counts_ho).float().mean().item() * 100

    print(f"\n베이스라인(평균값만 예측) heldout MSE(log)={baseline_loss:.4f}, round_acc={baseline_acc:.1f}%")
    print("\nprobe 완료.")


if __name__ == "__main__":
    main()
