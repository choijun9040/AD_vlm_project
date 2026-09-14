"""
Option C 사전 검증 (2단계) — sum-pooling density head 프로브
==============================================================
1차 프로브(probe_vision_count.py)는 patch feature를 mean pooling한 뒤 linear
회귀 → corr 0.22, held-out MSE는 baseline보다 나빴음(과적합). 하지만 mean pooling은
"몇 개인가"의 공간 정보를 뭉개므로 count에 불리한 집계다.

이 스크립트는 Option C의 핵심 설계(density map)를 최소 비용으로 검증한다:
backbone은 완전히 frozen인 채, vision encoder layer 23의 patch feature에
patch별 density head(softplus(Linear(1280,1)))를 달아 patch마다 비음수 "개수
기여도"를 예측하고, 전체 patch 합(sum)을 예측 개수로 쓴다. crowd counting에서
density map을 적분하면 인원수가 되는 것과 동일한 구조 — patch를 뭉개기 "전에"
세고 나서 합친다는 점이 mean 프로브와의 결정적 차이.

주의: 단순 "sum 후 linear"은 linear의 가법성 때문에 mean 프로브와 수학적으로
동치이므로 의미가 없다. 반드시 patch별 softplus(비선형)를 sum 이전에 걸어야
sum-pooling density가 mean과 구분된다.

판단: sum-pooling이 mean pooling(corr 0.22, held-out MSE > baseline)보다 유의미하게
나으면 → Option C 정식 구현 근거 강화. 비슷하거나 나쁘면 → vision feature 자체의
count 신호가 빈약하다는 뜻(집계 방식 문제가 아님) → Option C도 밑빠진 독.

실행:
    python scripts/probe_vision_count_sum.py
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
SPATIAL_LAYER_IDX = 23
VISION_HIDDEN = 1280
DRIVELM_TRAIN_JSON = "data/QA_dataset_nus/v1_0_train_nus.json"
NUSCENESQA_TRAIN_JSON = "data/nuscenes_qa/NuScenes_train_questions.json"

N_SAMPLES = 800          # per-patch feature를 들고 있어야 해서 1차(2000)보다 줄임
HELDOUT_N = 150
SEED = 42

captured = {}


def hook(module, inp, output):
    captured["feat"] = output.detach()


def parse_count(answer_str):
    m = re.match(r"^-?\d+$", answer_str.strip())
    return float(m.group()) if m else None


class DensityHead(nn.Module):
    """patch별 비음수 density 예측 후 sum → 예측 개수."""

    def __init__(self, hidden):
        super().__init__()
        self.lin = nn.Linear(hidden, 1)

    def forward(self, patch_feat, mask):
        # patch_feat: (B, P, hidden), mask: (B, P) — 패딩된 patch는 0
        density = F.softplus(self.lin(patch_feat)).squeeze(-1)  # (B, P) 비음수
        density = density * mask
        return density.sum(dim=1)  # (B,) 예측 개수


def main():
    device = "cuda"
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[모델 로드] {STUDENT_BASE} (frozen)")
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
    print(f"per-patch feature 추출 대상: {len(pool)}개")

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
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"].to(torch.bfloat16),
                image_grid_thw=inputs["image_grid_thw"],
            )
            pf = captured["feat"]
            if pf.dim() == 3:
                pf = pf.squeeze(0)
            feats.append(pf.to(torch.bfloat16).cpu())  # (P, hidden) ragged 가능
            counts.append(parse_count(s["answer"]))
            if (i + 1) % 200 == 0:
                print(f"  추출 {i + 1}/{len(pool)}", flush=True)

    h.remove()

    patch_counts = [f.shape[0] for f in feats]
    P_max = max(patch_counts)
    print(f"  patch 수: min={min(patch_counts)} max={P_max} (동일하면 고정 grid)")

    # 패딩 + 마스크로 (N, P_max, hidden) 스택
    N = len(feats)
    X = torch.zeros(N, P_max, VISION_HIDDEN, dtype=torch.bfloat16)
    mask = torch.zeros(N, P_max, dtype=torch.float32)
    for i, f in enumerate(feats):
        X[i, :f.shape[0]] = f
        mask[i, :f.shape[0]] = 1.0
    counts_t = torch.tensor(counts, dtype=torch.float32)

    perm = torch.randperm(N)
    ho_idx, tr_idx = perm[:HELDOUT_N], perm[HELDOUT_N:]

    Xtr = X[tr_idx].float().to(device); mtr = mask[tr_idx].to(device); ytr = counts_t[tr_idx].to(device)
    Xho = X[ho_idx].float().to(device); mho = mask[ho_idx].to(device); yho = counts_t[ho_idx].to(device)

    head = DensityHead(VISION_HIDDEN).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-3)

    print(f"\n{'=' * 60}\nsum-pooling density head 학습 (train={len(tr_idx)}, held-out={len(ho_idx)})\n"
          f"target=raw count (density 합), loss=SmoothL1\n{'=' * 60}\n")

    for epoch in range(300):
        opt.zero_grad()
        pred = head(Xtr, mtr)
        loss = F.smooth_l1_loss(pred, ytr)
        loss.backward()
        opt.step()

        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                pred_ho = head(Xho, mho)
                loss_ho = F.smooth_l1_loss(pred_ho, yho).item()
                mse_ho = F.mse_loss(pred_ho, yho).item()
                acc = (torch.round(pred_ho).clamp(min=0) == yho).float().mean().item() * 100
                corr = torch.corrcoef(torch.stack([pred_ho, yho]))[0, 1].item()
            print(
                f"  epoch {epoch + 1}/300 | train_smoothl1={loss.item():.4f} "
                f"heldout_smoothl1={loss_ho:.4f} heldout_MSE={mse_ho:.4f} "
                f"round_acc={acc:.1f}% corr={corr:.3f}",
                flush=True,
            )

    # 베이스라인: 그냥 train 평균 개수만 예측
    mean_pred = ytr.mean()
    baseline_mse = F.mse_loss(mean_pred.expand_as(yho), yho).item()
    baseline_acc = (torch.round(mean_pred).clamp(min=0) == yho).float().mean().item() * 100

    print(f"\n베이스라인(평균 개수만 예측) heldout MSE(raw count)={baseline_mse:.4f}, round_acc={baseline_acc:.1f}%")
    print("\n[비교] 1차 mean-pooling 프로브: corr=0.22 (log space), held-out MSE는 baseline보다 나빴음(과적합)")
    print("[해석] sum density의 corr/MSE가 mean보다 확실히 좋으면 → 집계 방식이 병목 → Option C 근거")
    print("       비슷하면 → vision feature 자체의 count 신호가 빈약 → 집계로는 못 살림")
    print("\nprobe 완료.")


if __name__ == "__main__":
    main()
