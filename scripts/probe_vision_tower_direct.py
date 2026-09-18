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
import time
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
        # **Phi-3-V 계열은 vision_config가 없다.** 대신 `img_processor.model_name`이
        # 표준 CLIP 저장소를 가리킨다(예: openai/clip-vit-large-patch14-336).
        # 그 config를 가져와 CLIPVisionModel을 세운다.
        ip = getattr(cfg, "img_processor", None)
        if isinstance(ip, dict) and ip.get("model_name"):
            from transformers import CLIPVisionConfig
            from transformers.models.clip.modeling_clip import CLIPVisionModel
            print(f"  vision_config 없음 — img_processor.model_name "
                  f"'{ip['model_name']}'에서 가져온다")
            vc = retry_429(lambda: CLIPVisionConfig.from_pretrained(ip["model_name"]))
            return CLIPVisionModel(vc), (archs[0] if archs else "Phi3V")
        raise RuntimeError("config에 vision_config가 없고 img_processor도 못 읽었다")
    table = {
        "Qwen2_5_VLForConditionalGeneration":
            ("qwen2_5_vl", "Qwen2_5_VisionTransformerPretrainedModel"),
        "Qwen2VLForConditionalGeneration":
            ("qwen2_vl", "Qwen2VisionTransformerPretrainedModel"),
        # **2026-09-18 확장.** 위험 구성 27개가 Qwen 계열에 치우쳐 있어, 실측 위험
        # 사례가 Qwen2.5-VL 하나로 몰렸다(선택 편향). 다른 계열을 재려고 넓혔다.
        "LlavaForConditionalGeneration": ("clip", "CLIPVisionModel"),
    }
    # **Pixtral은 arch가 LlavaForConditionalGeneration인데 타워가 CLIP이 아니다.**
    # HF가 Pixtral을 Llava 클래스로 감싸므로 arch만 보면 CLIPVisionModel을 세워
    # 조용히 틀린다. `vision_config.model_type`으로 먼저 가른다.
    vtype = getattr(vcfg, "model_type", None)
    if vtype == "pixtral":
        from transformers.models.pixtral.modeling_pixtral import PixtralVisionModel
        return PixtralVisionModel(vcfg), f"{archs[0] if archs else '?'}/pixtral"

    for a in archs:
        if a in table:
            mod, cls = table[a]
            m = __import__(f"transformers.models.{mod}.modeling_{mod}",
                           fromlist=[cls])
            return getattr(m, cls)(vcfg), a
    # 원격 코드 계열 — 저장소가 들고 다니는 모델링 파일에서 비전 클래스만 가져온다.
    # 전체 모델을 세우지 않는다(LLM이 양자화돼 있어 세울 수도 없다).
    remote = {
        "InternVLChatModel": ("modeling_intern_vit", "InternVisionModel"),
        "Phi3VForCausalLM":  ("modeling_phi3_v", "Phi3ImageEmbedding"),
    }
    for a in archs:
        if a in remote:
            fname, cls = remote[a]
            return _build_from_remote(cfg, fname, cls, vcfg), a
    raise RuntimeError(f"지원하지 않는 architectures: {archs}")


def _build_from_remote(cfg, fname, clsname, vcfg):
    """저장소의 모델링 파일을 내려받아 비전 클래스만 인스턴스화한다.

    `trust_remote_code`로 전체 모델을 세우면 양자화된 LLM까지 건드리게 되므로,
    **파일 하나만 import**해서 비전 클래스를 꺼낸다. 원격 코드를 실행하는 것은
    같으므로 잘 알려진 저장소에만 쓴다.
    """
    # 모델링 파일을 단독 exec하면 **상대 import가 깨진다**
    # (`from .configuration_intern_vit import ...`). transformers의 동적 모듈
    # 헬퍼가 패키지 맥락을 만들어 주므로 그것을 쓴다.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    repo = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", "")
    cls = get_class_from_dynamic_module(f"{fname}.{clsname}", repo)
    return cls(vcfg)


def fetch_vision_state(repo):
    """비전 키가 든 샤드만 내려받아 접두사를 벗긴 state_dict를 만든다."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    try:
        idx = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))
        wmap = idx["weight_map"]
        vis_keys = [k for k in wmap
                    if ".visual." in k or k.startswith("visual.")
                    or k.startswith("vision_model.") or k.startswith("vision_tower.")
                    or ".vision_tower." in k or "img_processor." in k]
        shards = sorted({wmap[k] for k in vis_keys})
        print(f"  비전 키 {len(vis_keys)}개 · 샤드 {len(shards)}/{len(set(wmap.values()))}개만 받는다")
    except Exception:
        shards, vis_keys = ["model.safetensors"], None
        print("  index 없음 — 단일 safetensors를 받는다")

    raw = {}
    for sh in shards:
        raw.update(load_file(hf_hub_download(repo, sh)))

    # **접두사는 계열마다 다르다 (2026-09-18 일반화).** 후보를 늘어놓고 **가장 많은
    # 키가 붙는 것**을 고른다. 하드코딩하면 계열이 늘 때마다 깨진다.
    cands = ("model.visual.", "visual.", "vision_model.", "vision_tower.",
             "model.vision_tower.", "model.vision_embed_tokens.img_processor.",
             "vision_embed_tokens.img_processor.")
    hit = {p: sum(1 for k in raw if k.startswith(p)) for p in cands}
    best = max(hit, key=hit.get)
    if hit[best] == 0:
        raise RuntimeError(f"비전 가중치를 찾지 못했다 — 접두사 후보 {list(cands)} "
                           f"중 일치 없음 (예시 키: {list(raw)[:3]})")
    print(f"  접두사 '{best}' — {hit[best]}개 키")
    sd = {k[len(best):]: v for k, v in raw.items() if k.startswith(best)}
    del raw
    gc.collect()
    q = [k for k in sd if any(t in k for t in QUANT_MARKERS)]
    if q:
        raise RuntimeError(f"비전 타워가 양자화되어 있다 ({len(q)}개 키, 예: {q[0]}) — "
                           "이 경로로는 잴 수 없다. 역양자화가 필요하다")
    return sd


def retry_429(fn, tries=5, base=20):
    """HF 허브가 429를 던지면 물러섰다 다시 친다.

    네트워크 사정은 결과가 아니다 — 재시도로 없앨 수 있는 실패를 "적재 실패"로
    기록하면 6장의 파편화 논거에 잡음이 섞인다.
    """
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            # huggingface_hub이 내부 재시도 후 **429를 LocalEntryNotFoundError나
            # OSError로 감싸서** 던진다. 문자열만 보면 놓치므로 예외 사슬까지 훑는다.
            chain, cur = [], e
            while cur is not None:
                chain.append(f"{type(cur).__name__}: {cur}")
                cur = cur.__cause__ or cur.__context__
            joined = " | ".join(chain)
            transient = ("429" in joined or "Too Many Requests" in joined
                         or "LocalEntryNotFoundError" in joined
                         or "couldn't connect" in joined)
            if not transient or i == tries - 1:
                raise
            wait = base * (2 ** i)
            print(f"  일시적 실패({type(e).__name__}) — {wait}초 후 재시도 "
                  f"({i+1}/{tries-1})", flush=True)
            time.sleep(wait)


def profile(repo, paths, max_pixels, dtype=torch.bfloat16, trust=False):
    from transformers import AutoConfig, AutoProcessor
    print(f"\n{'='*70}\n[{repo}]\n{'='*70}", flush=True)
    cfg = retry_429(lambda: AutoConfig.from_pretrained(
        repo, trust_remote_code=trust))
    tower, arch = build_tower(cfg)
    print(f"  타워: {type(tower).__name__}  (architectures={arch})")

    sd = retry_429(lambda: fetch_vision_state(repo))
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
    # **프로세서 실패가 타워 측정을 막아서는 안 된다.** Phi-3.5는 타워 적재가
    # 391/391로 통과했는데 저장소 프로세서가 TypeError로 죽었다. 전처리는
    # 이미지 → 픽셀 텐서일 뿐이므로, 실패하면 비전 인코더가 실제로 쓰는
    # CLIP 프로세서로 대체하고 **그 사실을 결과에 남긴다.**
    # **이미지 프로세서만 쓴다.** 텍스트 쪽은 한 번도 안 쓰는데, 전체 프로세서를
    # 부르면 그쪽 결함에 걸려 넘어진다 — LLaVA는 `patch_size=None`으로,
    # Phi-3.5는 원격 프로세서 코드로 죽었다. 둘 다 타워 적재는 391/391 통과였다.
    from transformers import AutoImageProcessor
    proc_src = repo
    try:
        proc = retry_429(lambda: AutoImageProcessor.from_pretrained(
            repo, trust_remote_code=trust))
    except Exception as e:
        ip = getattr(cfg, "img_processor", None)
        fallback = (ip or {}).get("model_name") if isinstance(ip, dict) else None
        if not fallback:
            raise
        print(f"  저장소 프로세서 실패({type(e).__name__}) — "
              f"'{fallback}' 프로세서로 대체한다", flush=True)
        proc = retry_429(lambda: AutoImageProcessor.from_pretrained(fallback))
        proc_src = fallback
    # **계열 일반화 (2026-09-18).** 이 루프는 Qwen 전용이었다 — 전체 프로세서,
    # `image_grid_thw`, `tower.blocks`, `tower(pv, grid)` 네 곳이 모두 그렇다.
    # CLIP 기반 타워(LLaVA·Phi-3-V)는 넷 다 다르므로 `headroom_guard`의
    # 계열 일반화 함수를 재사용한다.
    from headroom_guard import get_blocks, tower_forward

    ipx = getattr(proc, "image_processor", proc)      # 전체/이미지 프로세서 둘 다 허용
    if max_pixels and hasattr(ipx, "max_pixels"):
        ipx.max_pixels = max_pixels
        ipx.min_pixels = 3136

    # **동적 여부는 속성이 아니라 실측으로 가른다 (2026-09-18 정정).**
    # 처음에는 `hasattr(ipx, "max_pixels")`로 판정했는데 그것은 Qwen 전용
    # 속성이라 **Pixtral을 고정으로 오판했다** — Pixtral은 `size.longest_edge`로
    # 종횡비를 보존하며 시퀀스가 자란다(320×180 → 240패치, 1600×900 → 2,304패치).
    # 크기가 다른 두 이미지를 넣어 **출력 형상이 달라지는지** 직접 본다.
    from PIL import Image as _I
    _probe = _I.open(paths[0]).convert("RGB")
    _a = ipx(images=[_probe.resize((320, 180))], return_tensors="pt")["pixel_values"]
    _b = ipx(images=[_probe.resize((1280, 720))], return_tensors="pt")["pixel_values"]
    _shape = lambda x: tuple(x.shape) if hasattr(x, "shape") else tuple(x[0][0].shape)
    dynamic = _shape(_a) != _shape(_b)
    print(f"  입력 형상: 320×180 → {_shape(_a)} · 1280×720 → {_shape(_b)}  "
          f"→ {'가변' if dynamic else '고정'}", flush=True)
    if not dynamic:
        # **CLIP 계열은 입력 해상도가 고정(보통 336x336)이다.** max_pixels가
        # 적용되지 않으므로 Qwen 계열과 **같은 조건이 아니다.** 결과에 남긴다.
        print(f"  ⚠ 고정 해상도 타워 — max_pixels {max_pixels:,}가 적용되지 않는다. "
              f"Qwen 계열과 조건이 다르므로 나란히 비교하지 말 것", flush=True)

    blocks = get_blocks(tower)
    maxes = []
    with torch.no_grad():
        for p in paths:
            enc = ipx(images=[Image.open(p).convert("RGB")], return_tensors="pt")
            pv = enc["pixel_values"].to("cuda", dtype)
            grid = (enc["image_grid_thw"].to("cuda")
                    if "image_grid_thw" in enc else None)
            rec = {}
            h = blocks[-1].register_forward_hook(
                lambda _m, _i, o: rec.__setitem__("t", (o[0] if isinstance(o, tuple) else o).detach()))
            if "image_sizes" in enc:
                # Pixtral 계열 — grid가 아니라 image_sizes를 요구한다
                tower(pv, enc["image_sizes"].to("cuda"))
            else:
                tower_forward(tower, pv, grid)
            h.remove()
            maxes.append(rec["t"].float().abs().max().item())

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
            "loader": "vision_only_strict", "processor_from": proc_src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--out", default="eval_results/vision_tower_direct.json")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--trust_remote_code", action="store_true",
                    help="저장소가 들고 다니는 모델링 코드를 실행한다. "
                         "InternVL·Phi-3-V 계열에 필요하며, **잘 알려진 저장소에만** 쓸 것")
    args = ap.parse_args()

    tok = build_token_to_images(DRIVELM_VAL)
    paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][: args.limit]
    print(f"이미지 {len(paths)}장, max_pixels={args.max_pixels:,}")

    out_path = Path(args.out)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for repo in args.repos:
        try:
            res[repo] = profile(repo, paths, args.max_pixels, trust=args.trust_remote_code)
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
