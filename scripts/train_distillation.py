"""
Stage 3: Driving-Aware Distillation (Resume 지원)
===================================================
Teacher(Qwen2.5-VL-7B LoRA FT) -> Student(Qwen2.5-VL-3B) 증류.

L_total = L_task + lambda_spatial * L_spatial + lambda_temporal * L_temporal

- L_spatial: teacher/student vision encoder 중간 block(둘 다 depth=32, hidden=1280으로
  동일 아키텍처)의 patch-level hidden state를 코사인 손실로 정렬해 "어디에 있는지"
  (공간 정보)를 보존. attention map 대신 hidden state를 쓰는 이유는 attention map
  기반 시도에서 스케일이 0.0004 수준까지 죽어 사실상 무의미했기 때문.
- L_temporal ("대안 2", TAM 없음): teacher에게만 과거 프레임을 포함한 K장 멀티프레임을
  입력해 시간적으로 맥락화된 이해를 만들고, student는 여전히 현재 프레임 1장만 보면서
  teacher의 답변 직전 hidden state(=이해)를 흉내내도록 함. Student는 배포 시에도
  프레임당 1회 추론만 하면 되는 구조를 그대로 유지.

output-level logit KD(L_hazard)는 이번 버전에서 완전히 제거됨.

Resume:
    CONFIG의 "resume_from"에 체크포인트 경로를 지정하면
    해당 LoRA 가중치와 temporal projection 가중치를 로드하고 이어서 학습합니다.
    (WeightedRandomSampler로 샘플 순서가 달라지므로 새로운 데이터를 봄)

실행:
    python scripts/train_distillation.py

체크포인트:
    checkpoints/student_distill_3/epoch_1/
    checkpoints/student_distill_3/step_1000/
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator

sys.path.append(str(Path(__file__).parent))
from dataloader import create_unified_dataloader

TEACHER_HIDDEN_SIZE = 3584  # Qwen2.5-VL-7B LLM hidden_size
STUDENT_HIDDEN_SIZE = 2048  # Qwen2.5-VL-3B LLM hidden_size


# =============================================================================
# 설정
# =============================================================================

CONFIG = {
    # 모델
    "teacher_base":       "Qwen/Qwen2.5-VL-7B-Instruct",
    "teacher_lora":       "checkpoints/teacher_lora/epoch_1",
    "student_base":       "Qwen/Qwen2.5-VL-3B-Instruct",
    "output_dir":         "checkpoints/student_distill_3",

    # Resume — None이면 처음부터, 경로 지정 시 해당 체크포인트 LoRA/projection 가중치 로드
    "resume_from":        None,

    # Student LoRA
    "lora_rank":          16,
    "lora_alpha":         32,
    "lora_dropout":       0.05,
    # vision attention(qkv/proj)도 LoRA 대상에 포함할지. gate/up/down_proj는 이름이 같아서
    # 이미 vision MLP에도 LoRA가 걸리고 있음(의도치 않은 기존 동작) — attention도 열어줘야
    # L_spatial의 그래디언트가 패치 간 정보 재분배(attention)까지 갱신할 수 있음.
    "lora_target_vision_attn": True,

    # 학습
    "num_epochs":         1,
    "batch_size":         2,
    "grad_accum_steps":   8,
    "learning_rate":      2e-5,
    "warmup_ratio":       0.05,

    # 손실 가중치
    "lambda_spatial":     1.0,
    "lambda_temporal":    1.0,

    # L_spatial: teacher/student 공통 vision block index (fullatt_block_indexes=[7,15,23,31]
    # 중 하나 — 전체attention이라 패치 간 전역 문맥 반영, merger 직전(31)보다 덜 추상화됨)
    "spatial_layer_idx":  23,

    # L_temporal: teacher에게 줄 프레임 수(현재 포함). 처음엔 2로 축소해서 스모크 테스트 권장.
    "temporal_k":         3,
    # teacher 멀티프레임 토큰화 max_length. K=3 기준 image_pad 토큰만 약 693개 소요됨을
    # 실측 확인 — truncation이 이미지 토큰을 조용히 잘라내면 즉시 크래시하므로 여유 있게 설정.
    "teacher_max_length": 1280,

    # 데이터
    "drivelm_json":       "data/QA_dataset_nus/v1_0_train_nus.json",
    "nuscenesqa_json":    "data/nuscenes_qa/NuScenes_train_questions.json",
    "hazard_labels_path": "data/hazard_labels.json",
    "drivelm_ratio":      0.4,
    "num_workers":        4,

    # 위험도 오버샘플링 — loss와는 독립적인 데이터 샘플링 축이라 그대로 유지
    # (baseline_2/kd_only_2와 데이터 분포를 동일하게 유지해 비교 가능성을 지킴)
    "hazard_oversample":      True,
    "hazard_oversample_beta": 0.5,

    # 로깅/저장
    "log_every":          50,
    "save_every_steps":   1000,
}


# =============================================================================
# 1. Loss 함수
# =============================================================================

class SpatialFeatureKDLoss(nn.Module):
    """
    L_spatial: 공간 정보 보존 Feature KD.

    teacher/student vision encoder는 depth=32, hidden_size=1280으로 아키텍처가 완전히
    동일(가중치 값은 다름)하므로, 중간 block의 patch-level hidden state를 projection
    없이 바로 코사인 손실로 정렬할 수 있다. attention map(스케일 0.0004 수준으로 죽음)
    대신 hidden state를 쓰는 이유가 이것.

    teacher는 L_temporal용으로 K개 프레임을 입력받으므로, vision block 출력에는 K개
    이미지의 패치가 모두 섞여 있다 — 이 중 "현재 프레임"(각 샘플의 마지막 이미지)에
    해당하는 패치만 offset으로 슬라이싱해서 student(단일 프레임)와 비교한다.
    """

    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self._captured = {}
        self._hooks = []

    def register_hooks(self, teacher_model, student_model):
        def make_hook(key):
            def hook(module, inp, out):
                self._captured[key] = out
            return hook

        # PeftModel 래핑 경로: base_model.model.visual.blocks[i]
        # (LoRA target_modules는 q/k/v/o/gate/up/down_proj[, qkv, proj]이고
        #  visual 모듈 자체는 이름이 바뀌지 않으므로 이 경로가 항상 유효함)
        teacher_blocks = teacher_model.base_model.model.visual.blocks
        student_blocks = student_model.base_model.model.visual.blocks
        self._hooks.append(
            teacher_blocks[self.layer_idx].register_forward_hook(make_hook("teacher"))
        )
        self._hooks.append(
            student_blocks[self.layer_idx].register_forward_hook(make_hook("student"))
        )

    def clear(self):
        self._captured.clear()

    def forward(self, teacher_image_grid_thw, student_image_grid_thw, temporal_k, batch_size):
        t_patches = self._captured["teacher"]   # (sum_b sum_k n_patch_bk, 1280)
        s_patches = self._captured["student"]   # (sum_b n_patch_b, 1280)

        # teacher: 이미지별 patch 수(merge 이전이라 t*h*w, spatial_merge_size로 안 나눔)
        t_patch_counts = (
            teacher_image_grid_thw[:, 0]
            * teacher_image_grid_thw[:, 1]
            * teacher_image_grid_thw[:, 2]
        )
        t_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=t_patch_counts.device),
            t_patch_counts.cumsum(0),
        ])

        # 샘플별 "현재 프레임"(K번째 이미지) 패치만 슬라이싱
        current_frame_patches = []
        for b in range(batch_size):
            last_idx = b * temporal_k + (temporal_k - 1)
            start, end = t_offsets[last_idx].item(), t_offsets[last_idx + 1].item()
            current_frame_patches.append(t_patches[start:end])

        s_patch_counts = (
            student_image_grid_thw[:, 0]
            * student_image_grid_thw[:, 1]
            * student_image_grid_thw[:, 2]
        ).tolist()
        student_frame_patches = torch.split(s_patches, s_patch_counts, dim=0)

        losses = []
        for t_p, s_p in zip(current_frame_patches, student_frame_patches):
            t_p = t_p.float()
            s_p = s_p.float()
            cos = F.cosine_similarity(t_p, s_p, dim=-1)  # (n_patch,)
            losses.append((1 - cos).mean())

        return torch.stack(losses).mean()


class TemporalContextKDLoss(nn.Module):
    """
    L_temporal ("대안 2", TAM 없음): teacher는 과거 K-1개 프레임 + 현재 프레임을 보고,
    student는 현재 프레임 1장만 본다. 둘 다 "답변 생성 직전" 위치(q_len-1)의 마지막
    decoder layer hidden state를 뽑아, teacher(시간적으로 맥락화된 이해) 쪽으로
    student(단일 프레임 이해)가 정렬되도록 학습 가능한 projection + 코사인 손실을 건다.
    """

    def __init__(self, teacher_hidden=TEACHER_HIDDEN_SIZE, student_hidden=STUDENT_HIDDEN_SIZE):
        super().__init__()
        self.proj = nn.Linear(student_hidden, teacher_hidden)

    def forward(self, teacher_hidden_states, student_hidden_states, teacher_q_len, student_q_len):
        # teacher_hidden_states/student_hidden_states = 각 모델의 hidden_states[-1] (B, T, H)
        idx_t = (teacher_q_len - 1).clamp(min=0)
        idx_s = (student_q_len - 1).clamp(min=0)
        batch_idx = torch.arange(teacher_hidden_states.shape[0], device=teacher_hidden_states.device)

        t_vec = teacher_hidden_states[batch_idx, idx_t]   # (B, teacher_hidden)
        s_vec = student_hidden_states[batch_idx, idx_s]   # (B, student_hidden)

        s_proj = self.proj(s_vec.float())
        cos = F.cosine_similarity(s_proj, t_vec.float(), dim=-1)
        return (1 - cos).mean()


# =============================================================================
# 2. 모델 빌드
# =============================================================================

def build_teacher(config):
    print(f"\n[Teacher] 로드: {config['teacher_base']}")

    processor = AutoProcessor.from_pretrained(
        config["teacher_base"],
        max_pixels=256 * 28 * 28,
        min_pixels=64  * 28 * 28,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["teacher_base"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    teacher = PeftModel.from_pretrained(base, config["teacher_lora"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("  Teacher frozen (전체 파라미터 학습 안 함)")
    return teacher, processor


def build_student(config):
    """
    resume_from이 있으면 해당 체크포인트에서 LoRA 어댑터 로드.
    없으면 student_base에서 새로 LoRA 적용.
    """
    print("  Student processor 다운로드 중...")
    AutoProcessor.from_pretrained(
        config["student_base"],
        max_pixels=256 * 28 * 28,
        min_pixels=64  * 28 * 28,
    )
    print("  Student processor 준비 완료")

    resume_from = config.get("resume_from")
    ckpt_path   = Path(resume_from) if resume_from else None

    if ckpt_path and ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        print(f"\n[Student] 체크포인트에서 로드: {ckpt_path}")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["student_base"],
            torch_dtype=torch.bfloat16,
        )
        student = PeftModel.from_pretrained(base, str(ckpt_path), is_trainable=True)
        print("  LoRA 어댑터 로드 완료 (이어서 학습)")
    else:
        print(f"\n[Student] 처음부터 로드: {config['student_base']}")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["student_base"],
            torch_dtype=torch.bfloat16,
        )
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        if config.get("lora_target_vision_attn", True):
            # vision attention(Qwen2_5_VLVisionAttention: qkv/proj) — LLM의 o_proj 등과
            # 이름이 겹치지 않아 정확히 vision attention만 추가로 학습 대상이 됨.
            # gate/up/down_proj는 이름이 같아 이미 vision MLP에도 LoRA가 걸려 있었음.
            target_modules += ["qkv", "proj"]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=target_modules,
            bias="none",
        )
        student = get_peft_model(base, lora_cfg)

    student.print_trainable_parameters()
    return student


# =============================================================================
# 3. 학습 루프
# =============================================================================

def train(config):

    accelerator = Accelerator(
        gradient_accumulation_steps=config["grad_accum_steps"],
        mixed_precision="bf16",
    )

    with open(config["hazard_labels_path"]) as f:
        hazard_labels = json.load(f)
    print(f"\n위험도 라벨 로드: {len(hazard_labels)}개 (오버샘플링 용도로만 사용, loss에는 반영 안 함)")

    teacher, processor = build_teacher(config)
    student = build_student(config)

    print("\nDataLoader 구성 중...")
    dataloader = create_unified_dataloader(
        drivelm_json=config["drivelm_json"],
        nuscenesqa_json=config["nuscenesqa_json"],
        hazard_labels=hazard_labels,
        processor=processor,
        drivelm_ratio=config["drivelm_ratio"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        hazard_oversample=config.get("hazard_oversample", False),
        hazard_oversample_beta=config.get("hazard_oversample_beta", 0.5),
        temporal_k=config["temporal_k"],
        teacher_max_length=config["teacher_max_length"],
    )

    spatial_criterion = SpatialFeatureKDLoss(layer_idx=config["spatial_layer_idx"])
    temporal_criterion = TemporalContextKDLoss(
        teacher_hidden=TEACHER_HIDDEN_SIZE, student_hidden=STUDENT_HIDDEN_SIZE,
    ).to(accelerator.device)

    resume_from = config.get("resume_from")
    if resume_from and (Path(resume_from) / "temporal_proj.pt").exists():
        state = torch.load(Path(resume_from) / "temporal_proj.pt", map_location="cpu")
        temporal_criterion.load_state_dict(state)
        print(f"  temporal_proj 가중치 로드 완료 (resume): {resume_from}")
    elif resume_from:
        print(f"  경고: {resume_from}에 temporal_proj.pt가 없어 projection을 새로 초기화합니다")

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, student.parameters()))
        + list(temporal_criterion.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    total_steps  = math.ceil(len(dataloader) / config["grad_accum_steps"]) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    student, temporal_criterion, optimizer, dataloader, scheduler = accelerator.prepare(
        student, temporal_criterion, optimizer, dataloader, scheduler
    )

    # vision block hook은 accelerator.prepare로 모델이 최종 배치된 뒤 등록
    spatial_criterion.register_hooks(teacher, student)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_info = f"체크포인트 {resume_from}" if resume_from else "처음부터"

    print("\n" + "=" * 60)
    print("Stage 3: Driving-Aware Distillation 시작")
    print(f"  Teacher:           {config['teacher_base']} + LoRA")
    print(f"  Student:           {config['student_base']} + LoRA")
    print(f"  시작:              {resume_info}")
    print(f"  epochs:            {config['num_epochs']}")
    print(f"  batch_size:        {config['batch_size']}")
    print(f"  effective_batch:   {config['batch_size'] * config['grad_accum_steps']}")
    print(f"  total_steps:       {total_steps}")
    print(f"  warmup_steps:      {warmup_steps}")
    print(f"  learning_rate:     {config['learning_rate']}")
    print(f"  lambda_spatial:    {config['lambda_spatial']}")
    print(f"  lambda_temporal:   {config['lambda_temporal']}")
    print(f"  spatial_layer_idx: {config['spatial_layer_idx']}")
    print(f"  temporal_k:        {config['temporal_k']}")
    print("=" * 60 + "\n")

    global_step = 0

    for epoch in range(1, config["num_epochs"] + 1):
        student.train()
        epoch_losses = {"total": 0.0, "task": 0.0, "spatial": 0.0, "temporal": 0.0}
        epoch_steps  = 0

        for step, batch in enumerate(dataloader):

            with accelerator.accumulate(student):

                input_ids      = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values   = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                labels         = batch["labels"]
                q_len          = batch["q_len"]

                teacher_input_ids      = batch["teacher_input_ids"]
                teacher_attention_mask = batch["teacher_attention_mask"]
                teacher_pixel_values   = batch["teacher_pixel_values"]
                teacher_image_grid_thw = batch["teacher_image_grid_thw"]
                teacher_q_len          = batch["teacher_q_len"]

                with torch.no_grad():
                    t_out = teacher(
                        input_ids=teacher_input_ids,
                        attention_mask=teacher_attention_mask,
                        pixel_values=teacher_pixel_values,
                        image_grid_thw=teacher_image_grid_thw,
                        output_hidden_states=True,
                    )

                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                    output_hidden_states=True,
                )

                loss_task = s_out.loss
                loss_spatial = spatial_criterion(
                    teacher_image_grid_thw=teacher_image_grid_thw,
                    student_image_grid_thw=image_grid_thw,
                    temporal_k=config["temporal_k"],
                    batch_size=config["batch_size"],
                )
                loss_temporal = temporal_criterion(
                    teacher_hidden_states=t_out.hidden_states[-1],
                    student_hidden_states=s_out.hidden_states[-1],
                    teacher_q_len=teacher_q_len,
                    student_q_len=q_len,
                )
                loss_total = (
                    loss_task
                    + config["lambda_spatial"] * loss_spatial
                    + config["lambda_temporal"] * loss_temporal
                )

                accelerator.backward(loss_total)
                spatial_criterion.clear()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(student.parameters()) + list(temporal_criterion.parameters()), 1.0
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_losses["total"]    += loss_total.item()
            epoch_losses["task"]     += loss_task.item()
            epoch_losses["spatial"]  += loss_spatial.item()
            epoch_losses["temporal"] += loss_temporal.item()
            epoch_steps += 1

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % config["log_every"] == 0:
                    avg    = {k: v / epoch_steps for k, v in epoch_losses.items()}
                    lr_now = scheduler.get_last_lr()[0]
                    drivelm_cnt = batch["source"].count("drivelm")
                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] "
                        f"[Epoch {epoch}/{config['num_epochs']}] "
                        f"step {global_step}/{total_steps} | "
                        f"loss {avg['total']:.4f} "
                        f"(task={avg['task']:.3f} spatial={avg['spatial']:.3f} temporal={avg['temporal']:.3f}) | "
                        f"lr {lr_now:.2e} | "
                        f"DriveLM {drivelm_cnt}/{config['batch_size']}"
                    )

                if global_step % config["save_every_steps"] == 0:
                    _save_checkpoint(
                        accelerator, student, temporal_criterion, processor,
                        output_dir, global_step, epoch,
                        epoch_losses, epoch_steps, config,
                    )

        avg = {k: v / max(1, epoch_steps) for k, v in epoch_losses.items()}
        print(
            f"\n[Epoch {epoch} 완료] "
            f"avg_loss={avg['total']:.4f} "
            f"(task={avg['task']:.3f} spatial={avg['spatial']:.3f} temporal={avg['temporal']:.3f})\n"
        )
        _save_checkpoint(
            accelerator, student, temporal_criterion, processor,
            output_dir, global_step, epoch,
            epoch_losses, epoch_steps, config,
            name=f"epoch_{epoch}",
        )

    print("\nStage 3 Distillation 완료!")
    last_ckpt = output_dir / ("epoch_" + str(config["num_epochs"]))
    print("최종 체크포인트: " + str(last_ckpt))


# =============================================================================
# 4. 체크포인트 저장
# =============================================================================

def _save_checkpoint(
    accelerator, model, temporal_criterion, processor,
    output_dir, global_step, epoch,
    epoch_losses, epoch_steps, config,
    name=None,
):
    save_name = name or f"step_{global_step}"
    save_path = Path(output_dir) / save_name
    save_path.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    unwrapped_temporal = accelerator.unwrap_model(temporal_criterion)
    torch.save(unwrapped_temporal.state_dict(), save_path / "temporal_proj.pt")

    avg = {k: v / max(1, epoch_steps) for k, v in epoch_losses.items()}
    torch.save({
        "global_step": global_step,
        "epoch":       epoch,
        "avg_loss":    avg,
        "config":      config,
    }, save_path / "training_state.pt")

    print(f"  체크포인트 저장: {save_path}")


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    train(CONFIG)