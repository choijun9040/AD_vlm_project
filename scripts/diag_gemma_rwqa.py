"""개정 29 — Gemma 4 비전 인코더 진단, RealWorldQA 765장(공식 처리기, max_soft_tokens 280·1120). 2026-11-04, 결과 보기 전 등록.
① bf16 연산 단위 측정(경로 (ii) — 풀러·표준화를 모델 형식에서): 최댓값·최악 여유·이미지별 초과 비율(예측 상실)
② fp16 실제 상실: 경로 (ii)(배포 — vLLM 수정 전·fp16 엔진)와 경로 (i)(공식 transformers — 풀러·표준화를 fp32로 승격). 출력 비유한이면 상실.
venv_gemma.
    python scripts/diag_gemma_rwqa.py --model_dir /workspace/gemma4_vision/gemma-4-26B-A4B-it --tokens 280 --out eval_results/cmp/diag_gemma_26B_280.json
"""
import argparse, json, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX
from probe_gemma4_vision import load_vision, forward_points

FP16_MAX, TARGET = 65504.0, 2.0


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tokens", type=int, default=280)
    ap.add_argument("--q", default="data/realworldqa/questions.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoImageProcessor
    ip = AutoImageProcessor.from_pretrained(a.model_dir, max_soft_tokens=a.tokens)
    rows = [json.loads(l) for l in open(a.q)]
    cache = []
    for r in rows:
        e = ip(images=[Image.open(r["image"]).convert("RGB")], return_tensors="pt")
        cache.append((e["pixel_values"], e["image_position_ids"]))
    res = {"model_dir": a.model_dir, "tokens": a.tokens, "n": len(cache)}
    import resume as RS
    RS.setup(a.out)                                      # 끊겨도 이어 하기
    prev = RS.load("fp16") or {}
    res.update(prev)
    # ② fp16 실제 상실 — 두 경로
    m16 = load_vision(a.model_dir, "float16")
    for path, up in (("ii", False), ("i", True)):
        if f"fp16_loss_path_{path}" in res:
            continue
        bad = []
        with torch.no_grad():
            for x in cache:
                bad.append(int(not torch.isfinite(forward_points(m16, x[0], x[1], upcast=up)["final"]).all()))
        res[f"fp16_loss_path_{path}"] = {"nonfinite": sum(bad), "rate": sum(bad) / len(bad), "per_image": bad}
        print(f"[fp16 경로 ({path})] 비유한 {sum(bad)}/{len(bad)} = {sum(bad) / len(bad):.3f}", flush=True)
        RS.save("fp16", {k: v for k, v in res.items() if k.startswith("fp16_loss_path_")})
    del m16; torch.cuda.empty_cache()
    # ① bf16 측정 — 경로 (ii)
    mb = load_vision(a.model_dir, "bfloat16")
    ops = HX.profile_ops(mb, cache, lambda m, x: forward_points(m, x[0], x[1], upcast=False)["final"], FP16_MAX, TARGET, tag="bf16")
    g = ops["global_max"]
    res["diagnosis_bf16"] = {"max_worst": g["max_worst"], "point": g["point"], "headroom_worst": g["headroom_worst"],
                             "any_exceed_rate_predicted_loss": ops.get("any_exceed_rate"), "first_exceed": (ops["first_exceed"] or {}).get("point")}
    print(f"[측정] 최대 {g['max_worst']:,.0f} @ {g['point']} · 최악 여유 {g['headroom_worst']:.3f} · 예측 상실 {ops.get('any_exceed_rate')}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    RS.done()
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
