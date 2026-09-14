"""
옵션 A 스모크 테스트 — 카운팅 전용 보조 회귀 head (generation 기준 검증판)
=========================================================================
1차 스모크 테스트에서 loss_count(log)는 내려갔지만 head 자체의 round-match
정확도는 800스텝 내내 5~17.5%를 오갈 뿐 뚜렷한 상승이 없었음. 더 중요한 문제:
head의 예측은 student의 실제 텍스트 생성 경로에 전혀 반영되지 않으므로,
"count 질문에 대한 실제 답변(exact-match)이 좋아지는지"는 지금까지 한 번도
측정하지 않았다.

이 버전은 --lambda_count 인자로 손실 항을 켜고 끌 수 있게 해서, 동일한 스텝 수·
동일한 데이터 순서(고정 시드)로 두 조건(1.0 vs 0.0)을 학습시키고, 학습 중간중간
held-out count 질문 세트에 실제 model.generate()를 돌려 exact-match accuracy
추이를 비교한다 — "count 손실을 추가하면 실제 생성 답변이 개선되는가"에 직접 답한다.

실행:
    python scripts/smoke_count_head.py --lambda_count 1.0 --out_tag with_count
    python scripts/smoke_count_head.py --lambda_count 0.0 --out_tag control
"""

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

sys.path.append(str(Path(__file__).parent))
from dataloader import NuScenesQADataset, build_token_to_images, make_collate_fn
from eval_utils import normalize_answer as normalize

SEED = 42

STUDENT_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
STUDENT_HIDDEN_SIZE = 2048
DRIVELM_TRAIN_JSON = "data/QA_dataset_nus/v1_0_train_nus.json"
NUSCENESQA_TRAIN_JSON = "data/nuscenes_qa/NuScenes_train_questions.json"

NUM_STEPS = 2000
BATCH_SIZE = 2
LOG_EVERY = 50
GEN_EVAL_EVERY = 250
GEN_EVAL_N = 80        # held-out 생성 평가 샘플 수
HELDOUT_N = 200         # 학습에서 제외하고 held-out으로 떼어둘 개수 (GEN_EVAL_N보다 넉넉히)
LR = 5e-5
LORA_R = 32
LORA_ALPHA = 64


class NuScenesQAWithAnswer(NuScenesQADataset):
    """tensor 모드는 answer 텍스트를 안 담으므로 count supervision을 위해 살려준다."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["answer"] = self.samples[idx]["answer"]
        return item


class CountHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_state):
        return self.head(hidden_state).squeeze(-1)


def parse_count(answer_str):
    m = re.match(r"^-?\d+$", answer_str.strip())
    return float(m.group()) if m else None


@torch.no_grad()
def generate_answer(model, processor, image_path, question, device, max_new_tokens=16):
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    out_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def run_generation_eval(model, processor, device, held_out_samples, token_to_images):
    model.eval()
    correct, total = 0, 0
    for s in held_out_samples:
        img_path = str(token_to_images[s["sample_token"]].get("CAM_FRONT", ""))
        pred = generate_answer(model, processor, img_path, s["question"], device)
        gt = s["answer"]
        total += 1
        if normalize(pred) == normalize(gt):
            correct += 1
    model.train()
    return correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_count", type=float, default=1.0)
    parser.add_argument("--out_tag", type=str, default="run")
    args = parser.parse_args()

    torch.manual_seed(SEED)

    device = "cuda"
    print(f"[모델 로드] {STUDENT_BASE} | lambda_count={args.lambda_count} | tag={args.out_tag}")
    processor = AutoProcessor.from_pretrained(
        STUDENT_BASE, max_pixels=256 * 28 * 28, min_pixels=64 * 28 * 28,
    )
    student = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        STUDENT_BASE, torch_dtype=torch.bfloat16,
    ).to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    student = get_peft_model(student, lora_config)
    student.print_trainable_parameters()

    count_head = CountHead(STUDENT_HIDDEN_SIZE).to(device).to(torch.bfloat16)

    print("[데이터 준비] NuScenes-QA train에서 count 타입만 필터링")
    token_to_images = build_token_to_images(DRIVELM_TRAIN_JSON)
    ds = NuScenesQAWithAnswer(
        json_path=NUSCENESQA_TRAIN_JSON,
        token_to_images=token_to_images,
        processor=processor,
    )
    all_count_samples = [s for s in ds.samples if s["template_type"] == "count"]
    all_count_samples = [s for s in all_count_samples if parse_count(s["answer"]) is not None]

    # 고정 시드로 held-out/train 분리 — 두 조건(lambda_count 1.0 vs 0.0)이 동일한 분리를 보게 함
    rng = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(all_count_samples), generator=rng).tolist()
    heldout_idx = set(perm[:HELDOUT_N])
    held_out_samples = [all_count_samples[i] for i in perm[:GEN_EVAL_N]]
    train_samples = [all_count_samples[i] for i in range(len(all_count_samples)) if i not in heldout_idx]

    ds.samples = train_samples
    print(f"train count 샘플: {len(train_samples)}, held-out(생성평가용): {len(held_out_samples)}")

    collate_fn = make_collate_fn(pad_token_id=processor.tokenizer.pad_token_id)
    gen = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                         num_workers=2, generator=gen)

    params = list(student.parameters()) + list(count_head.parameters())
    optimizer = torch.optim.AdamW([p for p in params if p.requires_grad], lr=LR)

    print(f"\n{'=' * 60}\n스모크 테스트 시작: L_task + {args.lambda_count} * L_count (log space)\n{'=' * 60}\n")

    # step 0 (학습 전) 기준선
    c, t = run_generation_eval(student, processor, device, held_out_samples, token_to_images)
    print(f"  [gen-eval] step 0 (학습 전 기준선) | exact-match acc={c/t*100:.1f}% ({c}/{t})", flush=True)

    student.train()
    step = 0
    running_task, running_count = 0.0, 0.0
    loader_iter = iter(loader)
    while step < NUM_STEPS:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values    = batch["pixel_values"].to(device).to(torch.bfloat16)
        image_grid_thw  = batch["image_grid_thw"].to(device)
        labels          = batch["labels"].to(device)
        q_len           = batch["q_len"].to(device)

        gt_counts = torch.tensor(
            [parse_count(a) for a in batch["answer"]], dtype=torch.bfloat16, device=device,
        )
        gt_counts_log = torch.log1p(gt_counts.float())

        out = student(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            labels=labels, output_hidden_states=True,
        )
        loss_task = out.loss

        last_hidden = out.hidden_states[-1]
        idx = (q_len - 1).clamp(min=0)
        h_at_pos = last_hidden[torch.arange(last_hidden.size(0)), idx]
        pred_count_log = count_head(h_at_pos)

        loss_count = F.mse_loss(pred_count_log.float(), gt_counts_log)
        loss_total = loss_task + args.lambda_count * loss_count

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([p for p in params if p.requires_grad], 1.0)
        optimizer.step()

        running_task += loss_task.item()
        running_count += loss_count.item()
        step += 1

        if step % LOG_EVERY == 0:
            print(
                f"  step {step}/{NUM_STEPS} | loss_task={running_task/LOG_EVERY:.4f} "
                f"loss_count(log)={running_count/LOG_EVERY:.4f}",
                flush=True,
            )
            running_task, running_count = 0.0, 0.0

        if step % GEN_EVAL_EVERY == 0:
            c, t = run_generation_eval(student, processor, device, held_out_samples, token_to_images)
            print(f"  [gen-eval] step {step} | exact-match acc={c/t*100:.1f}% ({c}/{t})", flush=True)

    c, t = run_generation_eval(student, processor, device, held_out_samples, token_to_images)
    print(f"\n[gen-eval] 최종 step {NUM_STEPS} | exact-match acc={c/t*100:.1f}% ({c}/{t})")
    print("\n스모크 테스트 완료.")


if __name__ == "__main__":
    main()
