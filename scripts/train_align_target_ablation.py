"""
B — 정렬 대상 ablation: 교사가 정말 필요한가
================================================
`L_align`은 교사(7B) 비전 특징에 코사인 정렬한다. 그런데 측정해보면 **교사의 같은 층
활성이 오히려 더 크다**(9,024 대 학생 130). 코사인 손실이라 방향만 제약하기 때문이다.

그러면 자연히 묻게 된다 — **교사가 아니라 아무 고정된 특징에 정렬해도 같은 효과가
나오는가?** 나온다면 이것은 "증류"의 효과가 아니라 **"중간층에 정렬 제약을 거는 것"
자체의 효과**이며, 논문의 프레이밍(증류 손실 설계)이 흔들린다. 심사에서 나올 질문이므로
먼저 잰다.

방법: `train_distillation.py`를 **수정하지 않고** import해서 CONFIG만 갈아끼운다
(λ 스윕이 그 스크립트를 호출 중이라 원본을 건드리지 않는다).

  - `teacher`  : 기존 조건 — 7B + LoRA (= λ 스윕의 λ=1.0 지점과 동일)
  - `base3b`   : 정렬 대상을 **미세조정 없는 3B 베이스**로 교체.
                 "더 나은 모델의 특징"이 필요한지 본다.

`spatial` 변형(λ_temporal=0, temporal_k=1), 5,000스텝, 시드 1234 — λ 스윕과 동일 조건이라
직접 비교된다.

실행:
    python scripts/train_align_target_ablation.py --target base3b
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import awq_compat  # noqa: E402
awq_compat.patch()

import torch  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

import train_distillation as TD  # noqa: E402
from train_distillation import CONFIG, VARIANTS, train  # noqa: E402

STUDENT_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"

# 정렬 대상 모델의 hidden size — TemporalContextKDLoss.proj 출력 차원에 쓰인다.
# main()에서 --target에 따라 갱신한다.
_TARGET_HIDDEN = 3584


# ---------------------------------------------------------------------------
# train_distillation.py를 **수정하지 않고** 두 곳만 우회한다 (λ 스윕이 그 파일을
# 호출 중이므로 원본을 건드릴 수 없다). 둘 다 동작을 바꾸지 않고 일반화만 한다.
#   1) build_teacher: teacher_lora=None이면 PeftModel 래핑을 건너뛴다
#   2) register_hooks: PeftModel 전용 경로(base_model.model.visual) 대신
#      래핑 여부와 무관하게 비전 타워를 찾는다
# ---------------------------------------------------------------------------

def _visual_of(m):
    for _ in range(4):
        if hasattr(m, "visual"):
            return m.visual
        nxt = getattr(m, "base_model", None) or getattr(m, "model", None)
        if nxt is None or nxt is m:
            break
        m = nxt
    raise AttributeError("비전 타워를 찾지 못했다")


def _build_teacher(config):
    print(f"\n[Align target] 로드: {config['teacher_base']} "
          f"(lora={config['teacher_lora']})")
    processor = AutoProcessor.from_pretrained(
        config["teacher_base"], max_pixels=256 * 28 * 28, min_pixels=64 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["teacher_base"], torch_dtype=torch.bfloat16, device_map="auto")
    if config.get("teacher_lora"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, config["teacher_lora"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print("  정렬 대상 frozen")
    return model, processor


def _register_hooks(self, teacher_model, student_model):
    def make_hook(key):
        def hook(module, inp, out):
            self._captured[key] = out
        return hook
    tb = _visual_of(teacher_model).blocks
    sb = _visual_of(student_model).blocks
    self._hooks.append(tb[self.layer_idx].register_forward_hook(make_hook("teacher")))
    self._hooks.append(sb[self.layer_idx].register_forward_hook(make_hook("student")))


#   3) TemporalContextKDLoss.proj: Linear(2048, 3584)로 7B 교사 크기에 고정돼 있다.
#      정렬 대상을 3B(hidden 2048)로 바꾸면 코사인 계산에서 차원이 안 맞아 죽는다.
#      spatial 변형은 λ_temporal=0이라 이 손실이 결과에 기여하지 않으므로, 대상 모델의
#      hidden size에 맞춰 proj를 만들어 계산만 통과시킨다(값은 0이 곱해져 버려진다).
_orig_tmp_init = TD.AsymmetricTemporalKDLoss.__init__


def _tmp_init(self, teacher_hidden=None, student_hidden=None):
    # 호출부가 TEACHER_HIDDEN_SIZE(3584)를 명시로 넘기므로 무시하고 대상 모델 크기를 쓴다
    sh = TD.STUDENT_HIDDEN_SIZE if student_hidden is None else student_hidden
    _orig_tmp_init(self, teacher_hidden=_TARGET_HIDDEN, student_hidden=sh)


TD.build_teacher = _build_teacher
TD.FeatureAlignKDLoss.register_hooks = _register_hooks
TD.AsymmetricTemporalKDLoss.__init__ = _tmp_init

TARGETS = {
    # 기존 조건 (재현 확인용) — λ 스윕 λ=1.0과 동일해야 한다
    "teacher": {"teacher_base": "Qwen/Qwen2.5-VL-7B-Instruct",
                "teacher_lora": "checkpoints/teacher_lora/epoch_1"},
    # 정렬 대상만 교체: 미세조정 없는 3B 베이스
    "base3b":  {"teacher_base": STUDENT_BASE, "teacher_lora": None},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(TARGETS), default="base3b")
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    global _TARGET_HIDDEN
    CONFIG.update(VARIANTS["spatial"])          # λ_spatial=1, λ_temporal=0, temporal_k=1
    CONFIG.update(TARGETS[args.target])
    _TARGET_HIDDEN = 2048 if args.target == "base3b" else 3584
    CONFIG["max_steps"] = args.max_steps
    CONFIG["seed"] = args.seed
    CONFIG["output_dir"] = args.output_dir or f"checkpoints/aligntgt_{args.target}"

    print(f"[정렬 대상 ablation] target={args.target}")
    print(f"  align 대상 모델: {CONFIG['teacher_base']} + {CONFIG['teacher_lora']}")
    print(f"  output_dir={CONFIG['output_dir']}  max_steps={CONFIG['max_steps']}  "
          f"seed={CONFIG['seed']}")
    print(f"  lambda_align={CONFIG['lambda_align']} "
          f"lambda_atc={CONFIG['lambda_atc']} temporal_k={CONFIG['temporal_k']}")

    train(CONFIG)


if __name__ == "__main__":
    main()
