"""개정 22 1단계 — RealWorldQA 765장에서 공식 Qwen2.5-VL-3B 비전 타워의 fp16 상실 진단. 2026-11-04, 결과 보기 전 등록.
bf16 연산 단위 측정(예측 상실 = 이미지별 초과 비율)과 fp16 무교정 타워 출력의 실제 비유한 비율. 처리기는 공식 기본값(max_pixels 12,845,056) — 모든 이미지를 원본 해상도로.

    /workspace/venv_main/bin/python scripts/diag_realworldqa.py --out eval_results/cmp/rwqa_diag.json
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat; awq_compat.patch()
import headroom_ext as HX
import headroom_guard as G

FP16_MAX, TARGET = 65504.0, 2.0


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--q", default="data/realworldqa/questions.jsonl")
    ap.add_argument("--max_pixels", type=int, default=12845056,
                    help="공식 처리기 기본값(preprocessor_config.json) — RealWorldQA 이미지(최대 216만 픽셀)를 모두 원본 해상도로. "
                         "2026-11-04 21:28 정정: 처음엔 DriveLM 값 1,440,000을 그대로 써 129/765장이 줄어든 채로 들어갔다")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoProcessor
    rows = [json.loads(l) for l in open(a.q)]
    paths = [r["image"] for r in rows]
    proc = AutoProcessor.from_pretrained(G.BASE_3B, max_pixels=a.max_pixels, min_pixels=3136)
    res = {"n": len(rows), "model": G.BASE_3B, "max_pixels": a.max_pixels, "attn_implementation": "기본(지정 안 함 — sdpa)"}
    import resume as RS
    RS.setup(a.out)                                      # 끊겨도 이어 하기 — fp16 단계 결과, bf16 측정 50장마다
    prev = RS.load("fp16")
    if prev is not None:
        res["fp16_actual_loss"] = prev
    else:
        res = _fp16_stage(res, proc, paths, G)
        RS.save("fp16", res["fp16_actual_loss"])
    bad = res["fp16_actual_loss"]["per_image"]
    print(f"[fp16 무교정] 비유한 {sum(bad)}/{len(bad)} = {sum(bad) / len(bad):.3f}", flush=True)
    _bf16_stage(res, proc, paths, G, a, RS)


def _fp16_stage(res, proc, paths, G):
    t16 = G.load_tower("none", G.BASE_3B, "float16", attn=None)
    c16 = G.build_cache(proc, paths, "float16", device="cpu")
    bad = []
    with torch.no_grad():
        for i, x in enumerate(c16):
            o = G.tower_forward(t16, x[0].to("cuda", torch.float16), x[1].to("cuda") if x[1] is not None else None)
            bad.append(int(not torch.isfinite(o).all()))
    res["fp16_actual_loss"] = {"nonfinite": sum(bad), "rate": sum(bad) / len(bad), "per_image": bad}
    del t16, c16; torch.cuda.empty_cache()
    return res


def _bf16_stage(res, proc, paths, G, a, RS):
    # 측정 — bf16 연산 단위
    tb = G.load_tower("none", G.BASE_3B, "bfloat16", attn=None)
    cb = G.build_cache(proc, paths, "bfloat16", device="cpu")
    fwd = lambda t, x: G.tower_forward(t, x[0].to("cuda", torch.bfloat16), x[1].to("cuda") if x[1] is not None else None)
    ops = HX.profile_ops(tb, cb, fwd, FP16_MAX, TARGET, tag="bf16")
    g = ops["global_max"]
    res["diagnosis_bf16"] = {"max_worst": g["max_worst"], "point": g["point"], "headroom_worst": g["headroom_worst"],
                             "any_exceed_rate_predicted_loss": ops.get("any_exceed_rate"), "first_exceed": (ops["first_exceed"] or {}).get("point")}
    print(f"[측정] 최대 {g['max_worst']:,.0f} @ {g['point']} · 최악 여유 {g['headroom_worst']:.3f} · 예측 상실 {ops.get('any_exceed_rate')}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    RS.done()
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
