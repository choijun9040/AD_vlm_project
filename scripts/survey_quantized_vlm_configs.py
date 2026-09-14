"""
공개 양자화 VLM 체크포인트의 config 패턴 조사 (6장 보강)
==========================================================
본 연구에서 INT4 평가가 1.39%로 붕괴한 원인은 양자화 자체가 아니라
(i) AWQ 저장본 config의 `torch_dtype: float16` 때문에 양자화 대상이 아닌 비전 타워가
fp16으로 실행된 것, (ii) 저장본에 딸린 processor의 해상도 기본값이 평가 해상도보다
훨씬 컸던 것이었다.

이것이 **우리 실행의 아티팩트인지, 도구 사슬의 체계적 성질인지**를 가리기 위해
HuggingFace에 공개된 양자화 VLM 체크포인트들의 config를 수집해 같은 패턴이
얼마나 흔한지 센다. 흔하다면 6장은 버그 보고가 아니라 방법론 경고가 된다.

GPU 불필요. 네트워크만 사용한다.

실행:
    python scripts/survey_quantized_vlm_configs.py
"""

import argparse
import json
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

HF_API = "https://huggingface.co/api/models"
RAW = "https://huggingface.co/{repo}/raw/main/{fname}"

# 비전-언어 모델로 판별할 architectures 키워드
VLM_HINTS = ("VL", "Vision", "LLaVA", "Llava", "InternVL", "Idefics", "PaliGemma",
             "MiniCPMV", "Phi3V", "Molmo", "Pixtral", "Gemma3", "SmolVLM")


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "config-survey/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def search_repos(queries, limit):
    seen, repos = set(), []
    for q in queries:
        url = f"{HF_API}?search={urllib.parse.quote(q)}&limit={limit}&sort=downloads&direction=-1"
        for m in (get_json(url) or []):
            rid = m.get("modelId") or m.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                repos.append(rid)
    return repos


def is_vlm(cfg):
    arch = " ".join(cfg.get("architectures") or [])
    return any(h in arch for h in VLM_HINTS) or "vision_config" in cfg


def inspect(repo):
    cfg = get_json(RAW.format(repo=repo, fname="config.json"))
    if not cfg:
        return None
    q = cfg.get("quantization_config")
    if not q or not is_vlm(cfg):
        return None

    pre = get_json(RAW.format(repo=repo, fname="preprocessor_config.json")) or {}
    skip = q.get("modules_to_not_convert") or q.get("modules_in_block_to_quantize") or []
    if isinstance(skip, str):
        skip = [skip]

    return {
        "repo": repo,
        "arch": (cfg.get("architectures") or ["?"])[0],
        "torch_dtype": cfg.get("torch_dtype"),
        "quant_method": q.get("quant_method"),
        "bits": q.get("bits"),
        # 비전 타워를 양자화에서 제외했는가 (제외하면 그 부분은 torch_dtype으로 실행된다)
        "vision_excluded": any("vis" in str(s).lower() for s in skip),
        "modules_to_not_convert": skip,
        "min_pixels": pre.get("min_pixels"),
        "max_pixels": pre.get("max_pixels"),
        "size": pre.get("size"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60, help="검색어별 후보 수")
    ap.add_argument("--out", default="eval_results/quantized_vlm_config_survey.json")
    args = ap.parse_args()

    queries = ["VL-AWQ", "VL-GPTQ", "VL-Instruct-AWQ", "InternVL AWQ", "llava awq",
               "vision AWQ", "VL-Instruct-GPTQ-Int4", "MiniCPM-V awq", "Qwen2.5-VL AWQ"]
    repos = search_repos(queries, args.limit)
    print(f"후보 저장소 {len(repos)}개 조회")

    rows = []
    for i, r in enumerate(repos):
        info = inspect(r)
        if info:
            rows.append(info)
            print(f"  [{len(rows):3d}] {r[:58]:58s} "
                  f"dtype={str(info['torch_dtype']):<9} "
                  f"{info['quant_method']}/{info['bits']}bit "
                  f"vision제외={info['vision_excluded']} "
                  f"max_pixels={info['max_pixels']}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(repos)} 확인", flush=True)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n_checked": len(repos), "rows": rows},
                              indent=2, ensure_ascii=False))

    n = len(rows)
    if n == 0:
        print("\n양자화 VLM 체크포인트를 찾지 못했다 — 검색어 조정 필요")
        return

    fp16 = sum(1 for r in rows if r["torch_dtype"] == "float16")
    vex = sum(1 for r in rows if r["vision_excluded"])
    both = sum(1 for r in rows if r["torch_dtype"] == "float16" and r["vision_excluded"])
    bigpix = sum(1 for r in rows if (r["max_pixels"] or 0) > 4_000_000)

    print("\n" + "=" * 74)
    print(f"양자화 VLM 체크포인트 {n}개 분석")
    print("-" * 74)
    print(f"  torch_dtype == float16                    : {fp16:3d}/{n} ({fp16/n*100:.0f}%)")
    print(f"  비전 타워를 양자화에서 제외               : {vex:3d}/{n} ({vex/n*100:.0f}%)")
    print(f"  **둘 다 (= 비전 타워가 fp16으로 실행)**   : {both:3d}/{n} ({both/n*100:.0f}%)")
    print(f"  processor max_pixels > 4M (사실상 무제한) : {bigpix:3d}/{n} ({bigpix/n*100:.0f}%)")
    print("=" * 74)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
