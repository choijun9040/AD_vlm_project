"""
NuScenes-QA val 평가 스크립트
=================================
DriveLM val(v1_1_val_nus_q_only.json)의 799개 키프레임과 매핑되는 NuScenes-QA val
샘플(약 11,309개, train과 scene 단위로 완전 분리된 held-out)에서 Student 체크포인트의
단답형 QA 정확도를 측정한다.

NuScenes-QA 정답은 전부 짧은 폐집합 문자열(yes/no, 객체명, 상태, 숫자)이므로
exact-match Accuracy를 전체 + template_type별(exist/object/status/count/comparison)로 계산한다.

두 종류의 체크포인트를 모두 받는다:
  - LoRA 어댑터 (`adapter_config.json` 존재) → base 모델에 PEFT로 얹어 로드
  - AWQ INT4 병합본 (`config.json`에 quantization_config 존재) → 그대로 로드
어느 쪽이든 processor는 항상 base에서 동일 설정(min/max_pixels)으로 만들어,
정밀도만 다르고 입력 해상도는 같도록 프로토콜을 고정한다.

실행:
    python scripts/eval_nuscenesqa_val.py --checkpoint checkpoints/student_baseline_v2/epoch_1
    python scripts/eval_nuscenesqa_val.py --checkpoint checkpoints_awq/student_baseline_v2 \
        --limit 30 --dump-predictions
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
import awq_compat
from dataloader import NuScenesQADataset, build_token_to_images
from eval_utils import normalize_answer as normalize

DRIVELM_VAL_JSON    = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
NUSCENESQA_VAL_JSON = "data/nuscenes_qa/NuScenes_val_questions.json"
STUDENT_BASE        = "Qwen/Qwen2.5-VL-3B-Instruct"

# 학습·평가 전 구간에서 고정된 해상도 설정 — 정밀도 비교의 공정성을 위해 절대 바꾸지 않는다
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 64 * 28 * 28


def detect_precision(checkpoint_path: str) -> str:
    """체크포인트 종류를 판별해 'lora_bf16' 또는 'int4_awq' 등을 돌려준다."""
    ckpt = Path(checkpoint_path)
    cfg = ckpt / "config.json"
    if cfg.exists():
        q = json.loads(cfg.read_text()).get("quantization_config")
        if q:
            return f"{q.get('bits', '?')}bit_{q.get('quant_method', 'unknown')}"
    if (ckpt / "adapter_config.json").exists():
        return "lora_bf16"
    raise FileNotFoundError(
        f"{checkpoint_path}: adapter_config.json도 quantization_config도 없다 — 체크포인트 경로 확인"
    )


def load_model(checkpoint_path: str, precision: str, processor_from_checkpoint: bool = False,
               dtype: str = "float16", max_pixels: int = None, min_pixels: int = None,
               lora_dtype: str = "bfloat16"):
    # PEFT의 LoRA dispatcher도 AWQ 여부를 확인하느라 awq를 import하므로,
    # 정밀도와 무관하게 스텁을 먼저 꽂아둔다
    awq_compat.patch()

    if processor_from_checkpoint:
        # 체크포인트에 저장된 processor를 그대로 — AWQ 저장본은 min/max_pixels가
        # 기본값(3136 / 12,845,056)이라 학습·평가 때보다 해상도가 훨씬 커진다.
        # 과거 INT4 실행이 이 경로였는지 확인하는 대조군용 옵션.
        processor = AutoProcessor.from_pretrained(checkpoint_path)
    else:
        processor = AutoProcessor.from_pretrained(
            STUDENT_BASE,
            max_pixels=max_pixels or MAX_PIXELS,
            min_pixels=min_pixels if min_pixels is not None else MIN_PIXELS,
        )
    ip = processor.image_processor
    print(f"[processor] min_pixels={ip.min_pixels} max_pixels={ip.max_pixels}")

    if precision == "lora_bf16":
        # **lora_dtype (2026-09-15 추가)** — LoRA 경로는 bf16 고정이었다.
        # 배포 조건은 fp16이므로, 원본 해상도 정확도를 재려면 fp16으로도
        # 돌 수 있어야 한다. 기본값은 기존 동작(bfloat16)을 유지한다.
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            STUDENT_BASE, torch_dtype=getattr(torch, lora_dtype),
        )
        model = PeftModel.from_pretrained(base, checkpoint_path).to("cuda")
    else:
        # AWQ 병합본 — 커널을 참조 구현으로 고정 (triton fallback이 이 환경에서 깨짐)
        awq_compat.use_reference_kernel()
        # 주의: AWQ 저장본의 config는 torch_dtype=float16이지만 vision tower는
        # 양자화 대상이 아니라 이 dtype 그대로 돈다. Qwen2.5-VL의 vision tower는
        # 중간 활성값이 7,000대까지 올라가 fp16 한계(65504)에 근접하므로,
        # 학습 때와 같은 bfloat16으로 돌릴지 여기서 고를 수 있게 해 둔다.
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path, torch_dtype=getattr(torch, dtype), device_map="cuda",
        )

    model.eval()
    return model, processor


@torch.no_grad()
def generate_answer(model, processor, image_path: str, question: str, max_new_tokens: int) -> dict:
    """생성 결과와 함께 종료 신호 진단에 필요한 정보를 같이 돌려준다."""
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
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]

    eos_ids = {processor.tokenizer.eos_token_id, 151645, 151643}
    n_gen = int(gen_ids.shape[0])
    stopped = bool(set(gen_ids.tolist()) & eos_ids)

    return {
        "pred":     processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip(),
        "pred_raw": processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=False)[0],
        "n_gen_tokens": n_gen,
        "emitted_eos":  stopped,          # EOS를 한 번이라도 뱉었는가 (종료 신호 붕괴 진단)
        "hit_cap":      n_gen >= max_new_tokens and not stopped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="예: checkpoints/student_baseline_v2/epoch_1 또는 checkpoints_awq/student_baseline_v2")
    parser.add_argument("--limit", type=int, default=None, help="디버깅/스모크 테스트용 샘플 수 제한")
    # **--stride (2026-09-15 추가).** --limit N은 파일 앞쪽 N개를 자르는데, 문항이
    # 장면별로 뭉쳐 있어 앞 1,000문항이 799 키프레임 중 **79장면만** 덮는다.
    # 문항 수가 1,000이어도 독립 표본은 79이므로 이항 신뢰구간이 지나치게 좁아지고,
    # 템플릿 분포도 치우친다(comparison 15.7% → 12.4%).
    # --stride K는 K개마다 하나씩 뽑아 전체 키프레임에 고르게 퍼진 부분집합을 만든다.
    parser.add_argument("--stride", type=int, default=None,
                        help="K개마다 1문항씩 추출 (전체 키프레임에 고르게 퍼진 부분집합)")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--out_dir", default="eval_results")
    parser.add_argument("--dump-predictions", "--dump_predictions", dest="dump_predictions",
                        action="store_true",
                        help="샘플별 예측을 JSONL로 저장 (특수토큰 포함 raw 출력·생성 길이·EOS 방출 여부 포함)")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16",
                        help="양자화 체크포인트를 로드할 dtype (비양자화 vision tower가 이 dtype으로 돈다)")
    parser.add_argument("--processor_from_checkpoint", action="store_true",
                        help="base 대신 체크포인트에 저장된 processor를 사용 (해상도 교란 재현용 대조군)")
    parser.add_argument("--max_pixels", type=int, default=None,
                        help="미지정이면 평가 해상도(256*28*28). 원본 해상도는 1440000")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--lora_dtype", choices=["float16", "bfloat16"], default="bfloat16",
                        help="LoRA 경로의 기반 모델 dtype. 배포 조건(fp16)을 재려면 float16")
    parser.add_argument("--tag", default=None, help="출력 파일명에 붙일 꼬리표")
    # **--vision_state (2026-09-15 추가).** headroom_guard fix가 저장한 교정된 비전
    # 타워 state_dict를 얹는다. 도구가 산출한 가중치를 **그대로** 배포 조건에서
    # 평가하기 위한 경로다 — 손으로 고른 배수가 아니라 도구의 출력을 검증한다.
    parser.add_argument("--vision_state", default=None,
                        help="교정된 비전 타워 state_dict(.pt) 경로")
    parser.add_argument("--print_samples", type=int, default=0,
                        help="처음 N개 예측을 stdout에도 출력")
    args = parser.parse_args()

    precision = detect_precision(args.checkpoint)
    print(f"[모델 로드] {args.checkpoint}  (precision={precision})")
    model, processor = load_model(args.checkpoint, precision, args.processor_from_checkpoint,
                                  args.dtype, args.max_pixels, args.min_pixels, args.lora_dtype)

    if args.vision_state:
        # **LoRA를 먼저 병합한다 (2026-09-15 실측으로 잡음).** headroom_guard는
        # merge_and_unload() 한 타워의 state_dict를 저장하므로 키가 평문
        # (`patch_embed.proj.weight`)인데, 여기 모델은 PeftModel이라 타워 키가
        # LoRA로 감싸여 있다(`patch_embed.proj.base_layer.weight`). 그대로 얹으면
        # 643개가 어긋난다 — strict 검사가 이를 잡았다. 배포 시에도 병합본을
        # 내보내므로 병합이 올바른 조건이기도 하다.
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
            print("[교정] LoRA 병합 후 교정 가중치를 얹는다")
        # 비전 타워를 찾는다. **속성 이름을 순차 적용하면 안 된다** (2026-09-15 실측):
        # `model` 다음에 `visual`을 보면 Qwen2_5_VLForConditionalGeneration의
        # `.model`(텍스트 모델)에 걸려 `embed_tokens.weight`가 나온다. 경로 후보를
        # 통째로 시도해 **끝까지 성립하는 것**만 쓴다.
        vt = None
        for path in (("visual",), ("model", "visual"),
                     ("base_model", "model", "visual"),
                     ("base_model", "model", "model", "visual")):
            o, ok = model, True
            for a in path:
                if not hasattr(o, a):
                    ok = False
                    break
                o = getattr(o, a)
            if ok and hasattr(o, "blocks"):
                vt = o
                print(f"[교정] 비전 타워 경로: model.{'.'.join(path)} "
                      f"({type(o).__name__}, blocks={len(o.blocks)})")
                break
        if vt is None:
            raise RuntimeError("비전 타워를 찾지 못했다 — --vision_state를 적용할 수 없다")
        sd = torch.load(args.vision_state, map_location="cpu")
        missing, unexpected = vt.load_state_dict(sd, strict=False)
        pnames = {n for n, _ in vt.named_parameters()}
        miss_param = [k for k in missing if k in pnames]
        if miss_param or unexpected:
            raise RuntimeError(
                f"교정 가중치 적재 불일치 — 빠진 파라미터 {len(miss_param)}개"
                f"{', 예: ' + miss_param[0] if miss_param else ''}, "
                f"남은 키 {len(unexpected)}개"
                f"{', 예: ' + unexpected[0] if unexpected else ''}")
        vt.to(next(model.parameters()).dtype)
        print(f"[교정] 비전 타워 가중치 교체: {args.vision_state} "
              f"(파라미터 {len(pnames)}개 전부 적재)")

    print("[데이터 준비] DriveLM val 이미지 매핑 구성 중...")
    token_to_images = build_token_to_images(DRIVELM_VAL_JSON)

    ds = NuScenesQADataset(
        json_path=NUSCENESQA_VAL_JSON,
        token_to_images=token_to_images,
        processor=None,  # raw 모드 - generate를 직접 수행하므로 학습용 토큰화 불필요
    )
    samples = ds.samples
    if args.stride:
        samples = samples[:: args.stride]
        scenes = len({x.get("sample_token") for x in samples})
        print(f"[부분집합] stride={args.stride} → {len(samples)}문항, "
              f"{scenes}개 키프레임 (전체 799)")
    if args.limit:
        samples = samples[: args.limit]
    print(f"평가 대상 샘플 수: {len(samples)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ckpt_tag = (Path(args.checkpoint).name if precision != "lora_bf16"
                else Path(args.checkpoint).parent.name + "_" + Path(args.checkpoint).name)
    if precision != "lora_bf16":
        ckpt_tag += "_" + precision
    if precision != "lora_bf16" and args.dtype != "float16":
        ckpt_tag += "_" + args.dtype
    if args.processor_from_checkpoint:
        ckpt_tag += "_ckptproc"
    if args.tag:                          # 해상도·dtype 조건을 파일명에 남긴다
        ckpt_tag += "_" + args.tag
    if args.limit:                       # 전체 실행 결과를 스모크 실행이 덮어쓰지 않도록
        ckpt_tag += f"_limit{args.limit}"

    dump_f = None
    if args.dump_predictions:
        dump_path = out_dir / f"{ckpt_tag}_predictions.jsonl"
        dump_f = dump_path.open("w")
        print(f"[예측 덤프] {dump_path}")

    correct_total, total = 0, 0
    correct_by_type = defaultdict(int)
    total_by_type   = defaultdict(int)
    n_no_eos, n_hit_cap, gen_len_sum = 0, 0, 0

    for i, s in enumerate(samples):
        img_path = str(token_to_images[s["sample_token"]].get("CAM_FRONT", ""))
        g = generate_answer(model, processor, img_path, s["question"], args.max_new_tokens)
        gt = s["answer"]
        is_correct = normalize(g["pred"]) == normalize(gt)

        total += 1
        total_by_type[s["template_type"]] += 1
        if is_correct:
            correct_total += 1
            correct_by_type[s["template_type"]] += 1
        if not g["emitted_eos"]:
            n_no_eos += 1
        if g["hit_cap"]:
            n_hit_cap += 1
        gen_len_sum += g["n_gen_tokens"]

        if dump_f:
            dump_f.write(json.dumps({
                "idx": i,
                "sample_token":  s["sample_token"],
                "template_type": s["template_type"],
                "question": s["question"],
                "gt": gt,
                **g,
                "correct": is_correct,
            }, ensure_ascii=False) + "\n")
            dump_f.flush()

        if i < args.print_samples:
            mark = "O" if is_correct else "X"
            print(f"  [{mark}] ({s['template_type']}) Q: {s['question']}")
            print(f"      GT  : {gt!r}")
            print(f"      PRED: {g['pred']!r}   (tokens={g['n_gen_tokens']}, eos={g['emitted_eos']})")
            print(f"      RAW : {g['pred_raw']!r}", flush=True)

        if (i + 1) % args.log_every == 0:
            print(f"  [{i + 1}/{len(samples)}] 누적 Accuracy: {correct_total / total * 100:.2f}%", flush=True)

    if dump_f:
        dump_f.close()

    print("\n" + "=" * 50)
    print(f"체크포인트: {args.checkpoint}  (precision={precision})")
    print(f"전체 Accuracy: {correct_total}/{total} = {correct_total / max(1,total) * 100:.2f}%")
    print("카테고리별 Accuracy:")
    for t in sorted(total_by_type):
        acc = correct_by_type[t] / total_by_type[t] * 100
        print(f"  {t:12s}: {correct_by_type[t]:5d}/{total_by_type[t]:5d} = {acc:.2f}%")
    print("-" * 50)
    print("생성 종료 신호 진단:")
    print(f"  EOS 미방출     : {n_no_eos}/{total} = {n_no_eos / max(1,total) * 100:.2f}%")
    print(f"  길이 상한 도달 : {n_hit_cap}/{total} = {n_hit_cap / max(1,total) * 100:.2f}%")
    print(f"  평균 생성 길이 : {gen_len_sum / max(1,total):.2f} 토큰 (상한 {args.max_new_tokens})")
    print("=" * 50)

    result = {
        "checkpoint": args.checkpoint,
        "precision": precision,
        "dtype": args.dtype if precision != "lora_bf16" else "bfloat16",
        "max_new_tokens": args.max_new_tokens,
        "total": total,
        "correct": correct_total,
        "accuracy": correct_total / max(1, total),
        "eos_diagnostics": {
            "no_eos": n_no_eos,
            "no_eos_rate": n_no_eos / max(1, total),
            "hit_cap": n_hit_cap,
            "hit_cap_rate": n_hit_cap / max(1, total),
            "mean_gen_tokens": gen_len_sum / max(1, total),
        },
        "by_type": {
            t: {
                "correct": correct_by_type[t],
                "total": total_by_type[t],
                "accuracy": correct_by_type[t] / total_by_type[t],
            }
            for t in total_by_type
        },
    }
    out_path = out_dir / f"{ckpt_tag}_nuscenesqa_val.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
