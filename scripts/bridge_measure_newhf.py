#!/usr/bin/env python3
"""최신 transformers 환경에서 여유를 재는 **독립** 스크립트 (다리 측정).

**왜 따로 쓰는가.** 본 저장소의 측정은 전부 `transformers` 4.49에서 나왔고
(§6 "transformers를 올리지 않기로 했다"), 그 버전은 `qwen3_vl`을 모른다.
조사 197개 중 **최대 집단인 Qwen3-VL 57개**를 재려면 새 버전이 필요한데,
그냥 재면 **한 논문에 두 버전의 수치가 섞인다.**

그래서 **다리 측정**을 한다 — 같은 이미지·같은 조건으로 **Qwen2.5-VL-3B를
두 버전에서 각각** 재고, 값이 일치하는지 확인한다. 일치하면 새 버전의
Qwen3-VL 값을 기존 표에 나란히 둘 수 있다. 불일치하면 **그 자체가 결과**이고
버전 간 비교가 불가능함을 적는다.

**기존 코드에 의존하지 않는다.** `headroom_guard`·`awq_compat`·`peft`를 import하면
구 환경의 가정이 따라 들어온다. 다리 측정은 **독립적으로 검증 가능해야** 하므로
필요한 것만 여기에 다시 쓴다.

실행 (새 venv에서):
    /venv/hf_new/bin/python scripts/bridge_measure_newhf.py \
        --repos Qwen/Qwen2.5-VL-3B-Instruct --paths eval_results/bridge_image_paths.json
"""
import argparse
import gc
import json
import re
from pathlib import Path

import torch
from PIL import Image

FP16_MAX = 65504.0
QUANT_MARKERS = ("qweight", "qzeros", "scales", "g_idx")
PREFIXES = ("model.visual.", "visual.", "model.vision_tower.", "vision_tower.")


def fetch_vision_state(repo):
    """비전 키가 든 샤드만 받아 접두사를 벗긴 state_dict."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    try:
        idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
        wmap = idx["weight_map"]
        vis = [k for k in wmap if ".visual." in k or k.startswith("visual.")
               or ".vision_tower." in k or k.startswith("vision_tower.")]
        shards = sorted({wmap[k] for k in vis})
        print(f"  비전 키 {len(vis)}개 · 샤드 {len(shards)}/{len(set(wmap.values()))}개")
    except Exception:
        shards = ["model.safetensors"]
        print("  index 없음 — 단일 파일")
    raw = {}
    for sh in shards:
        raw.update(load_file(hf_hub_download(repo, sh)))
    hit = {p: sum(1 for k in raw if k.startswith(p)) for p in PREFIXES}
    best = max(hit, key=hit.get)
    if hit[best] == 0:
        raise RuntimeError(f"비전 가중치 없음 — 키 예시 {list(raw)[:3]}")
    print(f"  접두사 '{best}' — {hit[best]}개")
    sd = {k[len(best):]: v for k, v in raw.items() if k.startswith(best)}
    del raw; gc.collect()
    q = [k for k in sd if any(m in k for m in QUANT_MARKERS)]
    if q:
        raise RuntimeError(f"비전 타워가 양자화돼 있다 ({len(q)}개 키) — 이 경로로는 못 잰다")
    return sd


def build_tower(cfg):
    import transformers as HF
    archs = getattr(cfg, "architectures", None) or []
    vcfg = getattr(cfg, "vision_config", None)
    if vcfg is None:
        raise RuntimeError("vision_config 없음")
    table = {
        "Qwen2_5_VLForConditionalGeneration": ("qwen2_5_vl", "Qwen2_5_VisionTransformerPretrainedModel"),
        "Qwen2VLForConditionalGeneration":    ("qwen2_vl",   "Qwen2VisionTransformerPretrainedModel"),
        "Qwen3VLForConditionalGeneration":    ("qwen3_vl",   "Qwen3VLVisionModel"),
        "Qwen3VLMoeForConditionalGeneration": ("qwen3_vl_moe", "Qwen3VLMoeVisionModel"),
    }
    for a in archs:
        if a in table:
            mod, cls = table[a]
            m = __import__(f"transformers.models.{mod}.modeling_{mod}", fromlist=[cls])
            return getattr(m, cls)(vcfg), a
    raise RuntimeError(f"지원 밖 architectures: {archs}")


def get_blocks(tower):
    for attr in ("blocks", "layers"):
        if hasattr(tower, attr):
            return getattr(tower, attr)
    for sub in ("encoder", "transformer"):
        if hasattr(tower, sub):
            return get_blocks(getattr(tower, sub))
    raise RuntimeError(f"블록 리스트를 못 찾았다: {type(tower).__name__}")


def profile(repo, paths, max_pixels, dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoImageProcessor
    print(f"\n{'='*70}\n[{repo}]\n{'='*70}", flush=True)
    cfg = AutoConfig.from_pretrained(repo)
    tower, arch = build_tower(cfg)
    print(f"  타워: {type(tower).__name__}  ({arch})")

    sd = fetch_vision_state(repo)
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    pnames = {n for n, _ in tower.named_parameters()}
    miss_p = [k for k in missing if k in pnames]
    if miss_p or unexpected:
        raise RuntimeError(f"적재 실패 — 빠진 파라미터 {len(miss_p)}개 "
                           f"{miss_p[:3]} · 남는 키 {len(unexpected)}개 {list(unexpected)[:3]}")
    print(f"  적재 검증 통과: 파라미터 {len(pnames)}개 전부")
    tower = tower.to("cuda", dtype).eval()

    ipx = AutoImageProcessor.from_pretrained(repo)
    if hasattr(ipx, "max_pixels"):
        ipx.max_pixels = max_pixels; ipx.min_pixels = 3136
    blocks = get_blocks(tower)

    maxes = []
    with torch.no_grad():
        for p in paths:
            enc = ipx(images=[Image.open(p).convert("RGB")], return_tensors="pt")
            pv = enc["pixel_values"].to("cuda", dtype)
            grid = enc["image_grid_thw"].to("cuda") if "image_grid_thw" in enc else None
            rec = {}
            h = blocks[-1].register_forward_hook(
                lambda _m, _i, o: rec.__setitem__("t", (o[0] if isinstance(o, tuple) else o).detach()))
            tower(pv, grid) if grid is not None else tower(pv)
            h.remove()
            maxes.append(rec["t"].float().abs().max().item())
    del tower; gc.collect(); torch.cuda.empty_cache()

    s = sorted(maxes)
    if any(not (v == v and v != float("inf")) for v in s):
        raise RuntimeError("bf16에서 비유한 값 — 측정으로 쓰지 않는다")
    import transformers
    return {"repo": repo, "arch": arch, "n_images": len(s),
            "transformers": transformers.__version__,
            "last_block_p50": s[len(s)//2], "last_block_worst": s[-1],
            "headroom_p50": FP16_MAX / s[len(s)//2],
            "headroom_p95": FP16_MAX / s[int(len(s)*0.95)],
            "headroom_worst": FP16_MAX / s[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--paths", default="eval_results/bridge_image_paths.json")
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--out", default="eval_results/bridge_newhf.json")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    paths = json.loads(Path(args.paths).read_text())
    import transformers
    print(f"transformers {transformers.__version__} · 이미지 {len(paths)}장 "
          f"· max_pixels {args.max_pixels:,}")

    out = Path(args.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for repo in args.repos:
        try:
            res[repo] = profile(repo, paths, args.max_pixels)
            r = res[repo]
            print(f"  마지막 블록 p50={r['last_block_p50']:,.0f}  "
                  f"여유 p50={r['headroom_p50']:.2f}배  최악={r['headroom_worst']:.2f}배", flush=True)
        except Exception as e:
            print(f"  실패: {type(e).__name__}: {str(e)[:160]}", flush=True)
            res[repo] = {"failed": f"{type(e).__name__}: {e}"}
        out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        if args.cleanup:
            import os, shutil
            cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            d = Path(cache) / "hub" / ("models--" + repo.replace("/", "--"))
            if d.exists():
                shutil.rmtree(d, ignore_errors=True); print(f"  캐시 정리: {d.name}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
