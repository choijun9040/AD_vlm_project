"""
가중치 분포 확인 — massive activation을 만드는 것이 활성인가 가중치인가
==========================================================================
fixA 진단에서 blk31의 곱(act(gate)×up)은 4,588인데 `down_proj` 출력이 63,200으로
**13.8배 증폭**된다는 것이 드러났다. 그런데 지금까지 우리는 **활성만** 재고
가중치 분포는 본 적이 없다.

만약 `down_proj.weight`의 특정 행 노름이 비정상적으로 크다면, 현상의 정체는
"활성 크기"가 아니라 **"가중치 이상치"**다. 그리고 `L_align`이 하는 일도
"활성을 누르는 것"이 아니라 **"그 행 노름을 줄이는 것"**일 수 있다.

LoRA가 비전 MLP의 gate/up/down_proj에도 걸려 있으므로(이름이 LLM과 같아 자동 적용)
**병합된 유효 가중치**를 봐야 한다. GPU 불필요 — CPU에서 병합해 읽는다.

실행:
    python scripts/probe_weight_distribution.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
VARIANTS = {
    "base_3b":             None,
    "student_baseline_v2": "checkpoints/student_baseline_v2/epoch_1",
    "student_kd_only_v4":  "checkpoints/student_kd_only_v4/epoch_1",
    "student_temporal":    "checkpoints/student_temporal/epoch_1",
    "student_spatial":     "checkpoints/student_spatial/epoch_1",
    "student_full":        "checkpoints/student_full/epoch_1",
}
# blk23 = L_align이 정렬하는 층, blk31 = fp16 붕괴가 나는 층
LAYERS = (23, 31)
PROJS = ("gate_proj", "up_proj", "down_proj")


def stats(w):
    """행(출력 채널)별 L2 노름의 분포."""
    n = w.float().norm(dim=1)            # [out_features]
    s, _ = n.sort(descending=True)
    k = n.numel()
    return {
        "max": float(s[0]),
        "p99": float(s[max(0, k // 100)]),
        "p50": float(s[k // 2]),
        "max_over_p50": float(s[0] / s[k // 2]),
        "top5_idx": [int(i) for i in n.topk(5).indices],
        "out_features": k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_results/weight_distribution.json")
    args = ap.parse_args()
    awq_compat.patch()

    results = {}
    for name, adapter in VARIANTS.items():
        print(f"\n[{name}] 로드 (CPU)", flush=True)
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
        if adapter:
            m = PeftModel.from_pretrained(m, adapter)
            m = m.merge_and_unload()     # LoRA를 병합해 유효 가중치를 본다
        blocks = m.visual.blocks
        r = {}
        for li in LAYERS:
            for pj in PROJS:
                w = getattr(blocks[li].mlp, pj).weight.data
                r[f"blk{li}.{pj}"] = stats(w)
        results[name] = r
        del m
        for li in LAYERS:
            s = r[f"blk{li}.down_proj"]
            print(f"   blk{li}.down_proj  max={s['max']:8.2f}  p50={s['p50']:6.2f}  "
                  f"max/p50={s['max_over_p50']:6.1f}  top행={s['top5_idx'][:3]}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n" + "=" * 92)
    for li in LAYERS:
        print(f"\n=== blk{li} 행 노름 (LoRA 병합 후) ===")
        print(f"{'모델':22s}" + "".join(f"{p.replace('_proj',''):>11}" for p in PROJS)
              + f"{'down max/p50':>14}{'down top행':>12}")
        print("-" * 92)
        for n in VARIANTS:
            r = results[n]
            cells = "".join(f"{r[f'blk{li}.{p}']['max']:>11.1f}" for p in PROJS)
            d = r[f"blk{li}.down_proj"]
            print(f"{n:22s}{cells}{d['max_over_p50']:>14.1f}{d['top5_idx'][0]:>12}")
    print("=" * 92)
    print("해석: down_proj의 max 행 노름이 모델 간 갈리면 현상은 '가중치 이상치'다.")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
