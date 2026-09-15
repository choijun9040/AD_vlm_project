"""
① 계열 일반성 — 다른 VLM의 비전 타워 표현 범위 여유
======================================================
"마지막 블록이 massive activation을 생성하고 fp16 여유가 빠듯하다"는 현상이
Qwen2.5-VL 고유인지 계열 일반인지 확인한다. 학습 없이 비전 타워만 forward한다.

대상 선정 근거:
  - `Qwen/Qwen2-VL-*-Instruct-AWQ` — 6장 config 조사에서 **위험 구성**(torch_dtype
    float16 + 비전 타워 양자화 제외)으로 분류된 **제작사 공식 저장소**다. 실제로 여유가
    없는지 확인하면 "위험 구성이 실제 위험인가"에 답할 수 있다.
  - 다른 계열(Idefics3 등) — 현상이 아키텍처 계열을 넘는지 확인.

디스크가 빠듯하므로 소형 모델만 쓴다. 로드 실패는 건너뛴다.

실행:
    python scripts/profile_other_vlm_families.py --limit 100
"""

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images
from profile_vision_activations import DRIVELM_VAL_JSON, FP16_MAX, percentile

# (표시명, HF repo, 비고)
TARGETS = [
    ("qwen2vl_2b_awq", "Qwen/Qwen2-VL-2B-Instruct-AWQ", "공식 위험 구성 (fp16 + 비전 제외)"),
    ("smolvlm_instruct", "HuggingFaceTB/SmolVLM-Instruct", "다른 계열 (Idefics3)"),
    # 위험 구성 추가 표본 — 디스크가 빠듯해 하나씩 받고 지운다(--cleanup)
    ("qwen25vl_3b_awq", "AngelSlim/Qwen2.5-VL-3B-Instruct-INT4-AWQ",
     "위험 구성 · 본 연구 학생과 동일 아키텍처"),
    ("qwen2vl_7b_awq", "Qwen/Qwen2-VL-7B-Instruct-AWQ", "위험 구성 · 공식"),
    ("llava15_7b_awq", "ybelkada/llava-1.5-7b-hf-awq", "위험 구성 · 다른 계열(LLaVA)"),
    # 2026-09-14 확대 — 위험 구성 27개 중 **디스크가 감당하는 것 전부**를 실측한다.
    # 30B/32B/72B/235B는 4bit로도 8GB를 넘어 제외했다(그 사실 자체를 논문에 명시).
    # 목적: "위험 구성 27개"라는 config 집계를 **실측 여유 표**로 격상해 C2·C3을 닫는다.
    ("qwen25vl_7b_awq", "AngelSlim/Qwen2.5-VL-7B-Instruct-AWQ",
     "위험 구성 · 본 연구 교사와 동일 아키텍처"),
    # 아래 둘은 저장소의 커스텀 코드를 실행해야 로드된다(trust_remote_code).
    # 비-Qwen 계열 표본이 SmolVLM 하나뿐이라 C2(계열 일반성)에 값져서 포함했다.
    # **공식 저장소가 아니므로 임의 코드 실행 위험이 있다** — 이 둘에만 국소 허용한다.
    ("internvl3_8b_awq", "ELVISIO/SenseNova-SI-InternVL3-8B-AWQ",
     "위험 구성 · 다른 계열(InternVL3) [remote_code]"),
    ("phi35v_awq", "Isotr0py/Phi-3.5-vision-instruct-AWQ",
     "위험 구성 · 다른 계열(Phi-3.5-vision) [remote_code]"),
    ("qwen2vl_7b_awq_pc", "p-christ/Qwen2-VL-7B-Instruct-AWQ",
     "위험 구성 · 공식 재배포본 (같은 가중치면 같은 값이 나와야 한다 — 절차 점검)"),
    # --- 대형 (디스크가 나면 자동 진행, 안 나면 자동 건너뜀) ---
    # 27개 중 남은 18개는 대부분 30B 이상이다. --min_free_gb 가드가 각 모델 직전에
    # 여유를 확인하고 부족하면 **이유를 남기고 건너뛴다**. 논문에는 "측정한 N개"와
    # "디스크로 측정하지 못한 M개"를 함께 적는다.
    ("qwen3vl_30b_awq", "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ", "위험 구성 · 대형"),
    ("qwen3vl_32b_awq", "QuantTrio/Qwen3-VL-32B-Instruct-AWQ", "위험 구성 · 대형"),
    ("qwen25vl_32b_awq", "AngelSlim/Qwen2.5-VL-32B-Instruct-AWQ", "위험 구성 · 대형"),
    ("qwen2vl_72b_awq", "Qwen/Qwen2-VL-72B-Instruct-AWQ", "위험 구성 · 공식 대형"),
]

VISION_ATTRS = ("visual", "vision_model", "vision_tower", "vision_encoder")
BLOCK_HINTS = ("Block", "Layer")


def check_weights_loaded(model, repo, cls_name, sample=24):
    """가중치가 실제로 로드됐는지 확인한다.

    **왜 필요한가 (2026-09-15 실측).** 새 transformers로 저장된 체크포인트는 키가
    `model.visual.*` / `model.language_model.*`인데 4.49는 `visual.*` / `model.layers.*`를
    기대한다. 이름이 안 맞으면 `from_pretrained`가 **경고만 찍고 랜덤 초기화된 모델을
    돌려준다** — 로드는 "성공"하고 비전 타워도 찾아지므로, 확인하지 않으면 **의미 없는
    난수를 활성값으로 측정하게 된다.**

    판정: 비전 타워 파라미터 표본의 통계가 정상 초기화 분포와 구별되지 않으면 의심한다.
    구체적으로 LayerNorm/RMSNorm weight는 사전학습되면 1.0에서 벗어나는데, 랜덤
    초기화면 정확히 1.0이다.
    """
    tower = find_vision_tower(model)
    norms = [(n, p) for n, p in tower.named_parameters()
             if p.dim() == 1 and ("norm" in n.lower() or "ln" in n.lower())]
    if not norms:
        return
    exact_one = sum(1 for _, p in norms[:sample]
                    if bool((p.detach().float() == 1.0).all()))
    if exact_one >= max(1, min(sample, len(norms)) // 2):
        raise RuntimeError(
            f"가중치 미로드 의심 — 비전 타워 norm 파라미터 {exact_one}개가 정확히 1.0이다. "
            f"체크포인트 키 규약이 이 transformers 버전과 다를 가능성이 높다 "
            f"({cls_name}, {repo}).")


def find_vision_tower(model):
    """계열마다 이름이 달라 후보 속성을 순회하며 찾는다."""
    seen, stack = set(), [model]
    while stack:
        m = stack.pop(0)
        if id(m) in seen:
            continue
        seen.add(id(m))
        for a in VISION_ATTRS:
            if hasattr(m, a):
                return getattr(m, a)
        for child in m.children():
            stack.append(child)
    raise AttributeError("비전 타워를 찾지 못했다")


def find_blocks(tower):
    """비전 타워 안에서 transformer 블록 리스트를 찾는다."""
    best = None
    for name, mod in tower.named_modules():
        if isinstance(mod, (nn.ModuleList, nn.Sequential)) and len(mod) >= 4:
            cls = type(mod[0]).__name__
            if any(h in cls for h in BLOCK_HINTS):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
    if best is None:
        raise AttributeError("블록 리스트를 찾지 못했다")
    return best


@torch.no_grad()
def profile(tag, repo, note, image_paths, max_pixels):
    print(f"\n{'=' * 70}\n[{tag}] {repo}\n  {note}\n{'=' * 70}", flush=True)
    from PIL import Image

    # AutoModel은 계열에 따라 **비전 타워가 없는 내부 모델**을 돌려준다
    # (예: Qwen2-VL은 Qwen2VLModel = 텍스트 부분만). 2026-09-15 실측: Qwen2.5-VL AWQ
    # 저장본 4개가 이 함정에 걸려 "비전 타워를 찾지 못했다"로 전부 실패했다 —
    # 앞의 Auto 클래스들이 등록돼 있지 않아 마지막 AutoModel이 텍스트 모델을 돌려준 것이다.
    # 그래서 **config.architectures의 정확한 클래스를 먼저** 시도한다.
    import transformers as HF
    from transformers import AutoConfig

    trc = "[remote_code]" in note        # 이 태그에만 국소 허용
    order, cfg = [], None
    try:
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=trc)
        # 일부 AWQ 재저장본은 **더 새 transformers의 중첩 config 구조**로 쓰여 있다.
        # 4.49의 Qwen2_5_VLConfig는 평면 구조(hidden_size가 최상위)인데 저장본에는
        # `text_config` dict가 추가로 들어 있고, `get_text_config()`가 그 dict를
        # 돌려주면 `GenerationConfig.from_model_config`가
        # "'dict' object has no attribute 'to_dict'"로 죽는다. 그러면 상위 로더가
        # 조용히 AutoModel(텍스트 전용)로 내려가 "비전 타워를 찾지 못했다"가 된다.
        #
        # 실측(AngelSlim/Qwen2.5-VL-3B-INT4-AWQ): 최상위 필드와 text_config의 값이
        # **완전히 같고** 정상 저장본과도 일치한다. 즉 중복이므로 **제거하면 된다.**
        # 값이 다르면 제거가 위험하므로 그때는 건드리지 않고 경고만 남긴다.
        raw = getattr(cfg, "text_config", None)
        if isinstance(raw, dict):
            keys = ("hidden_size", "num_hidden_layers", "num_attention_heads",
                    "intermediate_size", "vocab_size")
            same = all(getattr(cfg, k, None) == raw.get(k) for k in keys if k in raw)
            if same:
                delattr(cfg, "text_config")
                print("  config 보정: 중복된 text_config dict 제거 "
                      "(최상위 필드와 값이 동일)", flush=True)
            else:
                print("  [경고] text_config가 dict인데 최상위와 값이 다르다 — "
                      "건드리지 않는다", flush=True)
        for arch in (getattr(cfg, "architectures", None) or []):
            if getattr(HF, arch, None) is not None:
                order.append(arch)
    except Exception as e:
        print(f"  config 읽기 실패(무시): {type(e).__name__}: {e}", flush=True)
    order += ["AutoModelForVision2Seq", "AutoModelForImageTextToText", "AutoModel"]

    model = None
    last_err = None
    for cls_name in order:
        cls = getattr(HF, cls_name, None)
        if cls is None:
            continue
        try:
            kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=trc)
            if cfg is not None and cls_name not in ("AutoModel",):
                kw["config"] = cfg          # 보정한 config를 쓴다
            model = cls.from_pretrained(repo, **kw)
            check_weights_loaded(model, repo, cls_name)
        except Exception as e:
            last_err = e
            print(f"  로더 {cls_name} 실패: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        # 로드는 됐어도 비전 타워가 없으면 다음 후보로 넘어간다(텍스트 전용 모델 방지)
        try:
            find_vision_tower(model)
        except AttributeError:
            print(f"  로더 {cls_name} → {type(model).__name__}: 비전 타워 없음, 다음 후보",
                  flush=True)
            last_err = AttributeError(f"{cls_name}가 비전 타워 없는 모델을 반환")
            del model
            model = None
            gc.collect()
            continue
        print(f"  로더: {cls_name} → {type(model).__name__}"
              + ("  (trust_remote_code)" if trc else ""), flush=True)
        break
    if model is None:
        raise RuntimeError(f"로드 실패: {last_err}")
    model = model.to("cuda").eval()
    tower = find_vision_tower(model)
    bname, blocks = find_blocks(tower)
    depth = len(blocks)
    print(f"  비전 타워 블록: {bname} (depth={depth}, {type(blocks[0]).__name__})", flush=True)

    proc = AutoProcessor.from_pretrained(repo, trust_remote_code=trc)
    if max_pixels and hasattr(proc, "image_processor") and hasattr(proc.image_processor, "max_pixels"):
        proc.image_processor.max_pixels = max_pixels
        proc.image_processor.min_pixels = 3136

    rec = {}

    def hook(i):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                rec[i] = t.detach()
        return fn

    hs = [blocks[i].register_forward_hook(hook(i)) for i in range(depth)]

    last_max, prev_max = [], []
    for p in image_paths:
        rec.clear()
        img = Image.open(p).convert("RGB")
        try:
            inputs = proc(text=["<image>"], images=[img], return_tensors="pt")
        except Exception:
            inputs = proc(images=[img], return_tensors="pt")
        inputs = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda"))
                  for k, v in inputs.items() if torch.is_tensor(v)}
        try:
            model(**inputs)
        except Exception:
            # 전체 forward가 실패하면(계열마다 이미지 토큰 표기가 달라 흔하다)
            # 비전 타워만 직접 호출한다. Qwen 계열은 grid_thw를 함께 요구한다.
            pv = inputs.get("pixel_values")
            if pv is None:
                raise
            grid = inputs.get("image_grid_thw")
            try:
                tower(pv, grid) if grid is not None else tower(pv)
            except TypeError:
                tower(pv)
        if depth - 1 in rec:
            last_max.append(rec[depth - 1].float().abs().max().item())
        if depth - 2 in rec:
            prev_max.append(rec[depth - 2].float().abs().max().item())

    for h in hs:
        h.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not last_max:
        raise RuntimeError("마지막 블록 출력을 수집하지 못했다")
    med = lambda v: percentile(sorted(v), 0.5)
    p50, p95 = med(last_max), percentile(sorted(last_max), 0.95)
    return {"repo": repo, "note": note, "depth": depth, "n_images": len(last_max),
            "last_block_p50": p50, "last_block_p95": p95,
            "prev_block_p50": med(prev_max) if prev_max else None,
            "headroom_p50": FP16_MAX / p50, "headroom_p95": FP16_MAX / p95}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_pixels", type=int, default=1440000, help="원본 해상도 조건")
    ap.add_argument("--out", default="eval_results/other_vlm_families_profile.json")
    ap.add_argument("--min_free_gb", type=float, default=0.0,
                    help="이 값보다 여유 디스크가 적으면 해당 모델을 건너뛴다(이유를 기록). "
                         "대형 위험 구성을 '자원이 나면 자동 진행'시키기 위한 가드다.")
    ap.add_argument("--cleanup", action="store_true",
                    help="모델별 프로파일 후 HF 캐시를 지운다 (디스크가 빠듯할 때)")
    args = ap.parse_args()

    awq_compat.patch()
    tok = build_token_to_images(DRIVELM_VAL_JSON)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.limit]
    print(f"대상 이미지 {len(paths)}장, max_pixels={args.max_pixels}")

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for tag, repo, note in TARGETS:
        # **성공한 것만** 건너뛴다. 실패·건너뜀 기록이 있으면 재시도한다
        # (로더를 고친 뒤 재실행할 수 있어야 한다).
        if "headroom_p50" in results.get(tag, {}):
            print(f"[{tag}] 이미 완료 — 건너뜀"); continue
        if args.min_free_gb > 0:
            import shutil as _sh
            free_gb = _sh.disk_usage("/workspace").free / 2**30
            if free_gb < args.min_free_gb:
                print(f"[{tag}] 디스크 부족으로 건너뜀 "
                      f"(여유 {free_gb:.1f} GB < 필요 {args.min_free_gb:.0f} GB)")
                results[tag] = {"repo": repo, "note": note,
                                "skipped": f"디스크 부족 (여유 {free_gb:.1f} GB)"}
                out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                continue
        try:
            results[tag] = profile(tag, repo, note, paths, args.max_pixels)
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            r = results[tag]
            print(f"  마지막 블록 p50={r['last_block_p50']:,.0f}  "
                  f"fp16 여유={r['headroom_p50']:.2f}배 (p95 기준 {r['headroom_p95']:.2f}배)", flush=True)
        except Exception as e:
            print(f"  [{tag}] 실패 — 건너뜀"); traceback.print_exc()
            # 실패도 기록한다 — 안 그러면 "측정 안 함"과 "측정 실패"를 구분할 수 없고,
            # 재실행 때 무엇이 남았는지도 알 수 없다.
            results[tag] = {"repo": repo, "note": note,
                            "failed": f"{type(e).__name__}: {str(e)[:200]}"}
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        finally:
            if args.cleanup:
                import shutil, os
                cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
                d = Path(cache) / "hub" / ("models--" + repo.replace("/", "--"))
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"  캐시 정리: {d.name}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'모델':22s}{'depth':>7}{'마지막 블록 p50':>17}{'fp16 여유':>11}")
    print("-" * 78)
    for tag, r in results.items():
        if "headroom_p50" not in r:
            continue
        print(f"{tag:22s}{r['depth']:>7}{r['last_block_p50']:>17,.0f}{r['headroom_p50']:>10.2f}배")
    print("=" * 78)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
