"""개정 42 ① — Qwen2.5-VL-3B 비전 인코더 블록별 최댓값, 진단 집합(DriveLM 학습 100장), bf16, 교정 전·잔차 재조정 뒤(c = 1/4). 2026-11-11, 결과 보기 전 등록.
headroom_guard.profile의 블록 출력(잔차 흐름) max|x|를 이미지별로 재고, 블록마다 100장 중 최댓값을 적는다. 그림용(원고 표 값은 바꾸지 않음). venv_main.

    /workspace/venv_main/bin/python scripts/qwen_block_profile.py --calib_q data/drivelm_official/train100.jsonl --out eval_results/cmp/env6/qwen_block_profile.json
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat; awq_compat.patch()
import headroom_guard as G, headroom_ext as HX


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--calib_q", required=True)
    ap.add_argument("--c", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoProcessor
    rows = [json.loads(l) for l in open(a.calib_q)]
    proc = AutoProcessor.from_pretrained(G.BASE_3B)                                  # 공식 기본값
    cache = G.build_cache(proc, [r["image"] for r in rows], "bfloat16", device="cuda")
    plan = json.load(open("eval_results/fixfam/qwen25_pretrained.json"))["residual"]
    res = {"n": len(rows), "c": a.c, "torch": torch.__version__}
    t = G.load_tower("none", G.BASE_3B, "bfloat16", attn=None)
    with torch.no_grad():
        pb, nan = G.profile(t, cache, "bfloat16", fmax_dtype="float16")
    res["before"] = {"per_block": pb, "nan_rate": nan}
    del t; torch.cuda.empty_cache()
    t32 = G.load_tower("none", G.BASE_3B, "float32", attn=None)
    with torch.no_grad():
        HX.apply_residual(t32, lambda m: m.blocks, plan["entry"], a.c, inplace=True)
        HX.apply_eps(t32, plan["eps_norms"], a.c)
    t = t32.to(torch.bfloat16)
    with torch.no_grad():
        pb, nan = G.profile(t, cache, "bfloat16", fmax_dtype="float16")
    res["after"] = {"per_block": pb, "nan_rate": nan}
    for k in ("before", "after"):
        m = [b["max_worst"] for b in res[k]["per_block"] if b["block"] != "tower_out"]
        print(f"[{k}] 블록 최댓값의 최대 {max(m):,.0f} @ 블록 {m.index(max(m))} · 상한 초과 블록 {sum(v > 65504 for v in m)} · 목표 초과 블록 {sum(v > 32752 for v in m)}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
