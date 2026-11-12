"""개정 21 — Qwen·Gemma의 c를 측정용 100장(DriveLM train 키프레임 CAM_FRONT, 시드 0, 평가 val 799장과 장면이 겹치지 않음)으로 정한다. 2026-11-04, 결과 보기 전 등록.
측정은 평가 조건과 같게: Qwen 원본 해상도(max_pixels 1,440,000) 비전 타워 bf16, Gemma 경로 (ii)(풀러·표준화를 모델 형식에서) bf16. 모든 연산의 최댓값 → c(목표 여유 2).

    venv_main:  python scripts/calib_train100.py qwen none --out eval_results/cmp/calib100_qwen_official.json
                python scripts/calib_train100.py qwen checkpoints/student_baseline_v2/epoch_1 --out eval_results/cmp/calib100_qwen_student.json
    venv_gemma: python scripts/calib_train100.py gemma /workspace/gemma4_vision/gemma-4-26B-A4B-it --tokens 280 --out ...
"""
import argparse, json, random, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX
from dataloader import build_token_to_images

TRAIN = "data/QA_dataset_nus/v1_0_train_nus.json"
FP16_MAX, TARGET, N, SEED = 65504.0, 2.0, 100, 0


def train_paths():
    tok = build_token_to_images(TRAIN)
    allp = sorted(str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v)
    return random.Random(SEED).sample(allp, N), len(allp)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("which", choices=["qwen", "gemma"])
    ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=280)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths, pool = train_paths()
    if a.which == "qwen":
        import awq_compat; awq_compat.patch()
        import headroom_guard as G
        from transformers import AutoProcessor
        tw = G.load_tower(a.model, G.BASE_3B, "bfloat16")
        cache = G.build_cache(AutoProcessor.from_pretrained(G.BASE_3B, max_pixels=1440000, min_pixels=3136), paths, "bfloat16", device="cpu")
        fwd = lambda t, x: G.tower_forward(t, x[0].to("cuda", torch.bfloat16), x[1].to("cuda") if x[1] is not None else None)
        model = tw
    else:
        from transformers import AutoImageProcessor
        from probe_gemma4_vision import load_vision, forward_points
        ip = AutoImageProcessor.from_pretrained(a.model, max_soft_tokens=a.tokens)
        cache = []
        for p in paths:
            enc = ip(images=[Image.open(p).convert("RGB")], return_tensors="pt")
            cache.append((enc["pixel_values"], enc["image_position_ids"]))
        model = load_vision(a.model, "bfloat16")
        fwd = lambda m, x: forward_points(m, x[0], x[1], upcast=False)["final"]
    ops = HX.profile_ops(model, cache, fwd, FP16_MAX, TARGET)
    g = ops["global_max"]
    c = HX.choose_c(g["max_worst"], FP16_MAX, TARGET)
    res = {"which": a.which, "model": a.model, "tokens": a.tokens if a.which == "gemma" else None, "n": len(paths), "pool": pool, "seed": SEED,
           "source": TRAIN, "max_worst": g["max_worst"], "point": g["point"], "any_exceed_rate": ops.get("any_exceed_rate"),
           "first_exceed": (ops["first_exceed"] or {}).get("point"), "c": c, "paths": paths}
    print(f"[측정용 {a.which} {Path(a.model).name}{' ' + str(a.tokens) if a.which == 'gemma' else ''}] 최대 {g['max_worst']:,.0f} @ {g['point']} · "
          f"초과 비율 {ops.get('any_exceed_rate')} · c = {c}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
