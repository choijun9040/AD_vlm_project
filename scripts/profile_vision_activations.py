"""
Vision tower 활성 크기 프로파일
=================================
증류 목적함수가 학생 vision 인코더의 활성 크기에 미치는 영향을 측정한다.
`scripts/probe_int4_nan.py`로 두 체크포인트에서 약 57배 차이(7,368 vs 128.5)를
확인한 것을, 5개 변형 + 베이스 + 교사에 대해 DriveLM val 전체 키프레임으로 확장한다.

두 가지를 동시에 잰다:
  1) **bf16 통과** — 표현 범위가 넉넉해 NaN 없이 참값을 얻는다. 층별 max|activation| 분포.
  2) **fp16 통과** — 실제 배포 dtype. 이미지 단위 NaN 발생률과 최초 발생 층.

LLM 디코더는 건드리지 않고 vision tower만 forward하므로 이미지당 비용이 작다.

실행:
    python scripts/profile_vision_activations.py                 # 전체 799 키프레임
    python scripts/profile_vision_activations.py --limit 50      # 스모크
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images

DRIVELM_VAL_JSON = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
BASE_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
BASE_7B = "Qwen/Qwen2.5-VL-7B-Instruct"

MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 64 * 28 * 28

FP16_MAX = 65504.0          # fp16 표현 한계 — 이 선을 넘으면 inf → 이후 전 층 NaN
KEY_LAYERS = (0, 8, 16, 24, 31)   # 이미지별로 따로 기록해 분포 그림에 쓸 층

# (표시명, base 모델, 어댑터 경로 or None, 손실 구성 메모)
MODELS = [
    ("base_3b",             BASE_3B, None,                                 "미세조정 없음 (출발점)"),
    ("student_baseline_v2", BASE_3B, "checkpoints/student_baseline_v2/epoch_1", "task CE only"),
    ("student_kd_only_v4",  BASE_3B, "checkpoints/student_kd_only_v4/epoch_1",  "task + 출력 KD"),
    ("student_temporal",    BASE_3B, "checkpoints/student_temporal/epoch_1",    "task + atc"),
    ("student_spatial",     BASE_3B, "checkpoints/student_spatial/epoch_1",     "task + align"),
    ("student_full",        BASE_3B, "checkpoints/student_full/epoch_1",        "task + align + atc"),
    ("teacher_7b",          BASE_7B, "checkpoints/teacher_lora/epoch_1",        "교사 (align의 정렬 대상)"),
    # 실제 배포 형태 — LoRA를 병합해 fp16으로 저장한 뒤 LLM만 INT4 양자화한 것.
    # vision tower는 양자화되지 않았지만 저장 dtype이 fp16이라 여기서 재는 값이 배포 실측치다.
    ("awq_baseline_v2", "checkpoints_awq/student_baseline_v2", None, "task CE only (병합·AWQ)"),
    ("awq_full",        "checkpoints_awq/student_full",        None, "task+align+atc (병합·AWQ)"),
]


def build_pixel_cache(processor, image_paths, workers):
    """이미지 전처리는 모델·dtype과 무관하므로 한 번만 하고 14회 재사용한다.

    느린(slow) image processor가 CPU 병목이라, 캐시 없이는 전체 실행이
    이미지 로딩에만 몇 시간을 쓴다. float32로 들고 있다가 통과 시점에 캐스팅한다.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(path):
        img = Image.open(path).convert("RGB")
        out = processor.image_processor(images=[img], return_tensors="pt")
        return out["pixel_values"], out["image_grid_thw"]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        cache = list(ex.map(one, image_paths))
    nbytes = sum(pv.element_size() * pv.nelement() for pv, _ in cache)
    print(f"전처리 캐시: {len(cache)}장, {nbytes / 2**30:.1f}GiB, {time.time() - t0:.0f}s",
          flush=True)
    return cache


def get_visual(model):
    """PeftModel/기본 모델 어느 쪽이든 vision tower를 찾아 돌려준다."""
    m = model
    for _ in range(4):
        if hasattr(m, "visual"):
            return m.visual
        m = getattr(m, "base_model", None) or getattr(m, "model", None)
        if m is None:
            break
    raise AttributeError("vision tower(visual)를 찾지 못했다")


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


@torch.no_grad()
def profile_model(name, base_id, adapter, pixel_cache, image_names, log_every, dump_f=None):
    print(f"\n{'=' * 66}\n[{name}] base={base_id} adapter={adapter}\n{'=' * 66}", flush=True)
    t0 = time.time()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to("cuda").eval()

    visual = get_visual(model)
    depth = len(visual.blocks)
    print(f"  vision depth={depth}  로드 {time.time() - t0:.0f}s", flush=True)

    # 이미지 전처리는 한 번만 하고 두 dtype 통과에 재사용
    result = {"depth": depth, "n_images": len(pixel_cache), "passes": {}}

    for dtype_name, dtype in [("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
        visual_d = get_visual(model.to(dtype))
        per_layer_max = [[] for _ in range(depth)]     # 이미지별 층 max|x|
        out_max, nan_images, first_nan_layer = [], 0, {}

        rec = {}

        def hook(idx):
            def fn(_m, _i, out):
                t = out[0] if isinstance(out, tuple) else out
                rec[idx] = t.detach()
            return fn

        handles = [visual_d.blocks[i].register_forward_hook(hook(i)) for i in range(depth)]

        t1 = time.time()
        for n, (pv_cpu, grid_cpu) in enumerate(pixel_cache):
            rec.clear()
            pv = pv_cpu.to("cuda", dtype, non_blocking=True)
            grid = grid_cpu.to("cuda", non_blocking=True)
            out = visual_d(pv, grid)

            nan_at = None
            for i in range(depth):
                t = rec[i].float()
                if torch.isnan(t).any() or torch.isinf(t).any():
                    per_layer_max[i].append(float("nan"))
                    if nan_at is None:
                        nan_at = i
                else:
                    per_layer_max[i].append(t.abs().max().item())

            o = out.float()
            if torch.isnan(o).any() or torch.isinf(o).any():
                nan_images += 1
                key = f"block{nan_at}" if nan_at is not None else "merger"
                first_nan_layer[key] = first_nan_layer.get(key, 0) + 1
                out_max.append(float("nan"))
            else:
                out_max.append(o.abs().max().item())

            if dump_f:
                dump_f.write(json.dumps({
                    "model": name, "dtype": dtype_name, "image": image_names[n],
                    **{f"blk{i}": per_layer_max[i][-1] for i in KEY_LAYERS if i < depth},
                    "visual_out": out_max[-1],
                    "nan": out_max[-1] != out_max[-1],
                }) + "\n")

            if log_every and (n + 1) % log_every == 0:
                print(f"    [{dtype_name}] {n + 1}/{len(pixel_cache)}  NaN 이미지 {nan_images}개",
                      flush=True)

        for h in handles:
            h.remove()

        def summarize(vals):
            clean = sorted(v for v in vals if v == v)          # NaN 제외
            return {
                "n_finite": len(clean),
                "p50": percentile(clean, 0.50),
                "p95": percentile(clean, 0.95),
                "p99": percentile(clean, 0.99),
                "max": clean[-1] if clean else None,
                # fp16 한계를 넘어선 이미지 비율 (NaN이 난 이미지는 넘은 것으로 계산)
                "over_fp16_rate": (sum(1 for v in vals if v != v or v > FP16_MAX)
                                   / max(1, len(vals))),
            }

        result["passes"][dtype_name] = {
            "by_layer": {str(i): summarize(per_layer_max[i]) for i in range(depth)},
            "visual_out": summarize(out_max),
            "nan_images": nan_images,
            "nan_rate": nan_images / max(1, len(pixel_cache)),
            "first_nan_layer": first_nan_layer,
            "seconds": round(time.time() - t1, 1),
        }
        s = result["passes"][dtype_name]
        print(f"  [{dtype_name}] blocks[{depth-1}] p50={s['by_layer'][str(depth-1)]['p50']}  "
              f"max={s['by_layer'][str(depth-1)]['max']}  "
              f"NaN {nan_images}/{len(pixel_cache)} ({s['nan_rate']*100:.1f}%)  "
              f"{s['seconds']}s", flush=True)

    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="이미지 수 제한 (스모크용)")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16, help="전처리 캐시 생성 병렬도")
    ap.add_argument("--max_pixels", type=int, default=MAX_PIXELS,
                    help="해상도 교란 검증용 — 기본값은 학습·평가에서 쓴 256*28*28")
    ap.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    ap.add_argument("--only", nargs="*", default=None, help="특정 모델만 (표시명)")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="MODELS에 없는 대상 추가: 'name=어댑터경로' (base는 3B 고정)")
    ap.add_argument("--out", default="eval_results/vision_activation_profile.json")
    ap.add_argument("--dump", default="eval_results/vision_activation_per_image.jsonl",
                    help="이미지별 층 max 기록 (분포 그림용). 빈 문자열이면 끔")
    args = ap.parse_args()

    awq_compat.patch()   # PEFT dispatcher가 awq를 import하므로 필요

    token_to_images = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in token_to_images.values() if "CAM_FRONT" in v]
    if args.limit:
        paths = paths[: args.limit]
    print(f"대상 이미지: {len(paths)}장 (DriveLM val CAM_FRONT)")

    processor = AutoProcessor.from_pretrained(BASE_3B, max_pixels=args.max_pixels,
                                              min_pixels=args.min_pixels)
    print(f'[processor] min_pixels={args.min_pixels} max_pixels={args.max_pixels}')
    pixel_cache = build_pixel_cache(processor, paths, args.workers)
    image_names = [Path(p).name for p in paths]

    targets = [m for m in MODELS if not args.only or m[0] in args.only]
    for spec in args.extra:
        name, _, path = spec.partition("=")
        targets.append((name, BASE_3B, path, f"추가 대상 ({path})"))
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.setdefault("_meta", {})
    results["_meta"].update({
        "n_images": len(paths), "max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
    })

    dump_f = open(args.dump, "a") if args.dump else None
    for name, base_id, adapter, note in targets:
        r = profile_model(name, base_id, adapter, pixel_cache, image_names, args.log_every, dump_f)
        r["loss"] = note
        r["adapter"] = adapter
        results[name] = r
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))   # 중간 저장
        if dump_f:
            dump_f.flush()
        print(f"  저장: {out_path}", flush=True)
    if dump_f:
        dump_f.close()

    # 요약표
    print("\n" + "=" * 78)
    print(f"{'모델':20s} {'손실':20s} {'blk24 p50':>10s} {'blk31 p50':>10s} "
          f"{'blk31 max':>10s} {'fp16 NaN율':>11s}")
    print("-" * 90)
    for name, *_ in targets:
        r = results.get(name)
        if not r:
            continue
        d = r["depth"]
        bl = r["passes"]["bfloat16"]["by_layer"]
        nan = r["passes"]["float16"]["nan_rate"] * 100
        print(f"{name:20s} {r['loss']:20s} {bl['24']['p50']:10.0f} "
              f"{bl[str(d-1)]['p50']:10.0f} {bl[str(d-1)]['max']:10.0f} {nan:10.1f}%")
    print("=" * 90)
    print(f"참고: fp16 표현 한계 = {FP16_MAX:.0f}")


if __name__ == "__main__":
    main()
