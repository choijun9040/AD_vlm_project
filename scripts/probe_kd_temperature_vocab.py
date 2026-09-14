"""
KD 온도 × 어휘 크기 스윕 (4장 보강)
=====================================
`kd_only_v2`의 EOS 붕괴를 진단하며 "Hinton식 온도 T=2~4는 ~1,000클래스 기준이라
152K 어휘에서는 종료 신호가 균등분포로 뭉개진다"고 **논증**했으나, 정작 어휘 크기를
바꿔가며 잰 적은 없다. 이 스크립트가 그 논증을 측정으로 바꾼다.

방법: 교사에게 질문+정답을 강제로 넣고, **정답 직후 위치**(EOS를 예측해야 하는 자리)의
로짓을 꺼낸다. 그 로짓을 상위 K개로 제한해 어휘 크기 K를 인위적으로 줄인 뒤
softmax(logits/T)를 계산하고, EOS의 확률·순위·2위와의 확률비를 기록한다.

  - K를 줄이면(= 작은 어휘) 같은 T에서도 EOS 확률이 유지되는가?
  - K를 키우면 같은 T에서 EOS 확률이 무너지는가?

두 질문에 답하면 "온도가 아니라 온도×어휘 크기가 변수"라는 진술이 성립한다.
학습은 하지 않는다 — 교사 forward만 돌린다.

실행:
    python scripts/probe_kd_temperature_vocab.py --n_samples 40
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images, DriveLMDataset

TEACHER_BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
TEACHER_LORA = "checkpoints/teacher_lora/epoch_1"
DRIVELM_TRAIN = "data/QA_dataset_nus/v1_0_train_nus.json"

EOS_ID = 151645          # <|im_end|>
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 64 * 28 * 28

VOCAB_SIZES = [1_000, 10_000, 100_000, None]      # None = 전체 어휘
TEMPERATURES = [1.0, 1.5, 2.0, 4.0, 8.0]


@torch.no_grad()
def eos_position_logits(model, processor, image_path, question, answer):
    """정답 직후(EOS를 예측해야 하는) 위치의 로짓을 돌려준다."""
    device = next(model.parameters()).device
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path},
                                     {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[Image.open(image_path).convert("RGB")],
                       return_tensors="pt").to(device)
    ids = inputs["input_ids"][0]

    # assistant 턴을 닫는 마지막 <|im_end|> 위치를 찾는다.
    eos_pos = (ids == EOS_ID).nonzero(as_tuple=True)[0]
    if len(eos_pos) == 0:
        return None
    target = int(eos_pos[-1])          # 이 자리에 EOS가 있어야 한다
    if target == 0:
        return None

    logits = model(**inputs).logits[0]
    return logits[target - 1].float()  # 그 직전 위치가 EOS를 예측한다


def analyze(logits, k, temperature):
    """상위 k개 어휘로 제한한 뒤 온도 T에서의 EOS 통계."""
    if k is None:
        sub, idx = logits, torch.arange(logits.numel(), device=logits.device)
    else:
        sub, idx = torch.topk(logits, k)
    pos = (idx == EOS_ID).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return None                     # EOS가 상위 k에 없다
    j = int(pos[0])

    p = torch.softmax(sub / temperature, dim=-1)
    p_eos = float(p[j])
    order = torch.argsort(p, descending=True)
    rank = int((order == j).nonzero(as_tuple=True)[0]) + 1
    top2 = torch.topk(p, 2).values
    runner_up = float(top2[1] if rank == 1 else top2[0])
    return {"p_eos": p_eos, "rank": rank,
            "ratio_to_best": p_eos / max(runner_up, 1e-30),
            "vocab": int(sub.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--out", default="eval_results/kd_temperature_vocab_sweep.json")
    args = ap.parse_args()

    awq_compat.patch()

    processor = AutoProcessor.from_pretrained(TEACHER_BASE, max_pixels=MAX_PIXELS,
                                              min_pixels=MIN_PIXELS)
    print("[교사 로드]", flush=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        TEACHER_BASE, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, TEACHER_LORA).to("cuda").eval()

    ds = DriveLMDataset(json_path=DRIVELM_TRAIN, processor=None)
    samples = ds.samples[: args.n_samples * 4]

    acc = {}    # (k, T) -> list of dict
    used = 0
    for s in samples:
        if used >= args.n_samples:
            break
        img = s.get("image_paths", {}).get("CAM_FRONT") or s.get("image_path")
        if not img or not Path(str(img)).exists():
            continue
        lg = eos_position_logits(model, processor, str(img), s["question"], s["answer"])
        if lg is None:
            continue
        for k in VOCAB_SIZES:
            for T in TEMPERATURES:
                r = analyze(lg, k, T)
                if r:
                    acc.setdefault(f"{k}|{T}", []).append(r)
        used += 1
        if used % 10 == 0:
            print(f"  {used}/{args.n_samples}", flush=True)

    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else None

    out = {"n_samples": used, "eos_id": EOS_ID,
           "vocab_sizes": [k or "full" for k in VOCAB_SIZES], "temperatures": TEMPERATURES,
           "results": {kk: {"p_eos_median": med([r["p_eos"] for r in vv]),
                            "rank_median": med([r["rank"] for r in vv]),
                            "ratio_to_best_median": med([r["ratio_to_best"] for r in vv]),
                            "vocab": vv[0]["vocab"], "n": len(vv)}
                       for kk, vv in acc.items()}}
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print(f"교사 EOS 확률 중앙값 — 정답 직후 위치, 샘플 {used}개")
    print(f"{'어휘 크기':>12} | " + " ".join(f"{'T='+str(T):>12}" for T in TEMPERATURES))
    print("-" * 78)
    for k in VOCAB_SIZES:
        cells = []
        for T in TEMPERATURES:
            r = out["results"].get(f"{k}|{T}")
            cells.append(f"{r['p_eos_median']:12.5f}" if r else f"{'-':>12}")
        print(f"{(str(k) if k else 'full'):>12} | " + " ".join(cells))
    print("=" * 78)
    print("해석: 같은 T에서 어휘가 커질수록 EOS 확률이 무너지면, 변수는 온도가 아니라 온도×어휘다.")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
