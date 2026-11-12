"""개정 23 — 공식 Qwen2.5-VL-3B 비전 타워에 (R)을 적용해 저장(Orin·TensorRT 내보내기용). 구성은 원고 공식 Qwen 구성(eval_results/fixfam/qwen25_pretrained.json —
자동 도구 구성과 같음, 개정 19), c는 인자로(개정 22 2단계 측정용 100문항 값). 형식은 checkpoints_fix/baseline_v2_residual_native.pt와 같다(fp16 state_dict + .eps.json). 2026-11-04.

    /workspace/venv_main/bin/python scripts/make_official_resid_state.py --c 0.25 --out checkpoints_fix/official_residual_rwqa.pt
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat; awq_compat.patch()
import headroom_ext as HX

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
PLAN = "eval_results/fixfam/qwen25_pretrained.json"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import Qwen2_5_VLForConditionalGeneration
    plan = json.load(open(PLAN))["residual"]
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
    v = m.visual
    HX.apply_residual(v, lambda t: t.blocks, plan["entry"], a.c, inplace=True)
    eps = HX.apply_eps(v, plan["eps_norms"], a.c)
    torch.save({k: t.half() for k, t in v.state_dict().items()}, a.out)
    Path(a.out + ".eps.json").write_text(json.dumps(eps, indent=1))
    print(f"저장: {a.out} (c = {a.c}, eps {len(eps)}개, 입구 {plan['entry']})")


if __name__ == "__main__":
    main()
