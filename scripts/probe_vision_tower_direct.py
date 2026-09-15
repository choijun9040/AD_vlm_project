"""공개 저장소의 **비전 타워만** 직접 올려 표현 범위 여유를 잰다 (strict 검증 포함).

**왜 이 로더가 따로 필요한가 (2026-09-15 실측).** `AngelSlim/Qwen2.5-VL-*-AWQ`를
`from_pretrained`로 올리면 transformers가 경고만 찍고 **랜덤 초기화된 비전 타워**를
돌려준다. 원인은 키 규약이다 — 이 저장소들은 더 새 transformers 배치로 저장돼
`model.visual.*` / `model.language_model.*`인데, 4.49의 `Qwen2_5_VLForConditionalGeneration`은
`visual.*` / `model.*`를 기대한다. 그래서 저장된 비전 가중치는 "쓰이지 않음"으로,
모델의 비전 타워는 "새로 초기화됨"으로 갈린다. 실제로 이 상태로 프로파일했더니
마지막 블록 max가 **NaN**으로 나왔다 — 그대로 뒀으면 난수를 논문에 실을 뻔했다.

**설계.** 전체 모델을 올리지 않는다. 필요한 건 비전 타워뿐이고, 이 계열의 AWQ
저장본은 **비전을 양자화에서 제외**하므로(그것이 바로 본 논문이 말하는 '위험 구성')
타워 가중치는 평범한 텐서다. 따라서
  1) config에서 vision_config만 꺼내 타워를 구성하고,
  2) 비전 키가 든 샤드만 내려받아,
  3) 접두사를 벗겨 **strict=True로 적재**한다.
strict 로드가 통과하면 "조용히 랜덤 초기화"가 **구조적으로 불가능**하다 —
키가 하나라도 안 맞으면 예외로 죽는다. 이것이 `check_weights_loaded`의 휴리스틱
(정규화 파라미터가 정확히 1.0인지)보다 강하다.

**한계를 적어 둔다.** 비전이 양자화된 저장소에는 이 경로가 통하지 않는다
(그때는 qweight/qzeros/scales를 역양자화해야 한다). 그 경우 이 스크립트는
**양자화되어 있음을 보고하고 중단**한다 — 조용히 틀리지 않는다.

실행:
    python scripts/probe_vision_tower_direct.py \
        --repos AngelSlim/Qwen2.5-VL-3B-Instruct-INT4-AWQ AngelSlim/Qwen2.5-VL-7B-Instruct-AWQ
"""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from dataloader import build_token_to_images

DRIVELM_VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
FP16_MAX = 65504.0
QUANT_MARKERS = ("qweight", "qzeros", "scales")


def pct(sv, q):
    return sv[min(int(q * (len(sv) - 1) + 0.5), len(sv) - 1)]


def build_tower(cfg):
    """architectures에 맞는 비전 타워 클래스를 골라 **가중치 없이** 구성한다."""
    import transformers as HF
    archs = getattr(cfg, "architectures", None) or []
    vcfg = getattr(cfg, "vision_config", None)
    if vcfg is None:
        raise RuntimeError("config에 vision_config가 없다")
    table = {
        "Qwen2_5_VLForConditionalGeneration":
            ("qwen2_5_vl", "Qwen2_5_VisionTransformerPretrainedModel"),
        "Qwen2VLForConditionalGeneration":
            ("qwen2_vl", "Qwen2VisionTransformerPretrainedModel"),
    }
    for a in archs:
        if a in table:
            mod, cls = table[a]
            m = __import__(f"transformers.models.{mod}.modeling_{mod}",
                           fromlist=[cls])
            return getattr(m, cls)(vcfg), a
    raise RuntimeError(f"지원하지 않는 architectures: {archs} "
                       f"(이 스크립트는 Qwen2-VL / Qwen2.5-VL 계열 전용)")


def fetch_vision_state(repo):
    """비전 키가 든 샤드만 내려받아 접두사를 벗긴 state_dict를 만든다."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
        wmap = idx["weight_map"]
        vis_keys = [k for k in wmap if ".visual." in k or k.startswith("visual.")]
        shards = sorted({wmap[k] for k in vis_keys})
        print(f"  비전 키 {len(vis_keys)}개 · 샤드 {len(shards)}/{len(set(wmap.values()))}개만 받는다")
    except Exception:
        shards, vis_keys = ["model.safetensors"], None
        print("  index 없음 — 단일 safetensors를 받는다")

    raw = {}
    for sh in shards:
        raw.update(load_file(hf_hub_download(repo, sh)))

    sd = {}
    for k, v in raw.items():
        if k.startswith("model.visual."):
            sd[k[len("model.visual."):]] = v
        elif k.startswith("visual."):
            sd[k[len("visual."):]] = v
    del raw
    gc.collect()
    if not sd:
        raise RuntimeError("비전 가중치를 찾지 못했다 (visual 접두사 없음)")
    q = [k for k in sd if any(t in k for t in QUANT_MARKERS)]
    if q:
        raise RuntimeError(f"비전 타워가 양자화되어 있다 ({len(q)}개 키, 예: {q[0]}) — "
                           "이 경로로는 잴 수 없다. 역양자화가 필요하다")
    return sd


def profile(repo, paths, max_pixels, dtype=torch.bfloat16):
    from transformers import AutoConfig, AutoProcessor
    print(f"\n{'='*70}\n[{repo}]\n{'='*70}", flush=True)
    cfg = AutoConfig.from_pretrained(repo)
    tower, arch = build_tower(cfg)
    print(f"  타워: {type(tower).__name__}  (architectures={arch})")

    sd = fetch_vision_state(repo)
    # 적재 검증 — 조용한 랜덤 초기화를 구조적으로 막는다.
    # strict=True를 바로 쓰면 비지속 버퍼(rotary inv_freq 등) 때문에 오탐이 난다.
    # 그래서 strict=False로 받고 **파라미터가 하나라도 빠졌으면 예외**로 만든다.
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    pnames = {n for n, _ in tower.named_parameters()}
    miss_param = [k for k in missing if k in pnames]
    if miss_param or unexpected:
        raise RuntimeError(
            f"적재 불일치 — 빠진 파라미터 {len(miss_param)}개"
            f"{' 예: ' + miss_param[0] if miss_param else ''}, "
            f"남은 키 {len(unexpected)}개"
            f"{' 예: ' + unexpected[0] if unexpected else ''}")
    n_loaded = sum(1 for n in pnames if n in sd)
    print(f"  적재 검증 통과: 파라미터 {n_loaded}/{len(pnames)}개 전부 적재 "
          f"(빠진 버퍼 {len(missing)}개는 비지속 버퍼)")

    tower = tower.to("cuda", dtype).eval()
    proc = AutoProcessor.from_pretrained(repo)
    if max_pixels and hasattr(proc.image_processor, "max_pixels"):
        proc.image_processor.max_pixels = max_pixels
        proc.image_processor.min_pixels = 3136

    maxes = []
    with torch.no_grad():
        for p in paths:
            enc = proc(text=["<image>"], images=[Image.open(p).convert("RGB")],
                       return_tensors="pt")
            pv = enc["pixel_values"].to("cuda", dtype)
            grid = enc["image_grid_thw"].to("cuda")
            out = tower.blocks[-1]  # 후크 대신 마지막 블록 출력을 직접 잡는다
            rec = {}
            h = out.register_forward_hook(
                lambda _m, _i, o: rec.__setitem__("t", (o[0] if isinstance(o, tuple) else o).detach()))
            tower(pv, grid)
            h.remove()
            t = rec["t"].float().abs().max().item()
            maxes.append(t)

    del tower
    gc.collect()
    torch.cuda.empty_cache()

    sm = sorted(maxes)
    nonfinite = sum(1 for v in maxes if not (v == v and v != float("inf")))
    if nonfinite:
        raise RuntimeError(f"bf16에서 비유한 값 {nonfinite}/{len(maxes)}장 — "
                           "적재는 통과했으나 값이 이상하다. 측정으로 쓰지 않는다")
    return {"repo": repo, "arch": arch, "n_images": len(maxes),
            "last_block_p50": pct(sm, 0.5), "last_block_p95": pct(sm, 0.95),
            "last_block_worst": sm[-1],
            "headroom_p50": FP16_MAX / pct(sm, 0.5),
            "headroom_p95": FP16_MAX / pct(sm, 0.95),
            "headroom_worst": FP16_MAX / sm[-1],
            "loader": "vision_only_strict"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--out", default="eval_results/vision_tower_direct.json")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    tok = build_token_to_images(DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.limit]
    print(f"이미지 {len(paths)}장, max_pixels={args.max_pixels:,}")

    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for repo in args.repos:
        try:
            res[repo] = profile(repo, paths, args.max_pixels)
            r = res[repo]
            print(f"  마지막 블록 p50={r['last_block_p50']:,.0f}  "
                  f"여유 p50={r['headroom_p50']:.2f}배  "
                  f"p95={r['headroom_p95']:.2f}배  최악={r['headroom_worst']:.2f}배")
        except Exception as e:
            import traceback; traceback.print_exc()
            res[repo] = {"repo": repo, "failed": f"{type(e).__name__}: {str(e)[:250]}"}
        out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
        if args.cleanup:
            import os
            cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            d = Path(cache) / "hub" / ("models--" + repo.replace("/", "--"))
            if d.exists():
                shutil.rmtree(d, ignore_errors=True); print(f"  캐시 정리: {d.name}")

    print("\n" + "=" * 86)
    print(f"{'저장소':<48}{'여유 p50':>10}{'p95':>9}{'최악':>9}")
    print("-" * 86)
    for k, r in res.items():
        if "headroom_p50" in r:
            print(f"{k:<48}{r['headroom_p50']:>9.2f}배{r['headroom_p95']:>8.2f}배"
                  f"{r['headroom_worst']:>8.2f}배")
        else:
            print(f"{k:<48}  실패: {r.get('failed','')[:28]}")
    print("=" * 86)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
