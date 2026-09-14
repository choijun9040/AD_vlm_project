"""
마지막 비전 블록 down_proj의 행 노름 이상치가 **구조적인지 이 체크포인트 고유인지** 본다.

5장에서 Qwen2.5-VL-3B의 `visual.blocks.31.mlp.down_proj` 849번 행이 중앙값의 9.3배
노름을 갖고, 이것이 오버플로의 증폭기임을 보였다. 그 행이 왜 생겼는지(사전학습 기원)는
답할 수 없지만, **다른 모델에도 같은 패턴이 있는지**는 가중치만 보면 알 수 있다.
있으면 구조적이고, 없으면 이 체크포인트 고유다 — 서술이 그만큼 정확해진다.

가중치만 읽으므로 GPU가 필요 없다. 디스크를 아끼려고 필요한 샤드 하나만 받고 지운다.
Qwen2-VL 계열은 비전 MLP 이름이 `fc1`/`fc2`라 두 규약을 모두 찾는다.

실행:
    python scripts/probe_downproj_outlier_across_models.py --cleanup
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

REPOS = [
    ("Qwen2.5-VL-3B (본 연구 대상)", "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("Qwen2.5-VL-7B (교사)", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("Qwen2-VL-2B (이전 세대)", "Qwen/Qwen2-VL-2B-Instruct"),
]
# 비전 MLP의 '출력쪽' 선형층 이름 후보
OUT_NAMES = ("down_proj", "fc2")


def find_last_vision_out(weight_map):
    best = None
    for k in weight_map:
        if ".weight" not in k:
            continue
        m = re.search(r"(?:visual|vision_tower|vision_model)\.(?:blocks|layers)\.(\d+)\.", k)
        if not m:
            continue
        if not any(f".{n}.weight" in k for n in OUT_NAMES):
            continue
        idx = int(m.group(1))
        if best is None or idx > best[0]:
            best = (idx, k)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true", help="샤드를 쓰고 바로 지운다")
    ap.add_argument("--out", default="eval_results/downproj_outlier_across_models.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    results, norm_vecs = {}, {}
    for label, repo in REPOS:
        print(f"\n=== {label} — {repo} ===", flush=True)
        try:
            idx_path = hf_hub_download(repo, "model.safetensors.index.json")
            wm = json.load(open(idx_path))["weight_map"]
            found = find_last_vision_out(wm)
            if not found:
                print("  비전 출력 선형층을 찾지 못했다 — 건너뜀")
                results[repo] = {"error": "키 없음"}
                continue
            blk, key = found
            shard = wm[key]
            print(f"  키: {key}  (블록 {blk}), 샤드 {shard}", flush=True)
            print("  샤드 내려받는 중 …", flush=True)
            sp = hf_hub_download(repo, shard)

            # numpy에는 bfloat16이 없어 framework="np"로는 못 읽는다(실측 TypeError).
            # torch로 읽고 float32로 올린다.
            import torch
            with safe_open(sp, framework="pt") as f:
                w = f.get_tensor(key)           # (out_features, in_features)
            norms = torch.linalg.norm(w.float(), dim=1).numpy()
            med = float(np.median(norms))
            order = np.argsort(norms)[::-1]
            top = [(int(i), float(norms[i]), float(norms[i] / med)) for i in order[:5]]

            print(f"  행 {len(norms)}개, 노름 중앙값 {med:.3f}")
            print("  상위 5행 (행번호, 노름, 중앙값 대비):")
            for i, n, r in top:
                print(f"    {i:>5}  {n:8.3f}  {r:6.2f}배")
            ratio = top[0][2]
            verdict = ("뚜렷한 이상치 있음" if ratio >= 4
                       else "약한 이상치" if ratio >= 2 else "이상치 없음")
            print(f"  판정: {verdict} (최대 {ratio:.2f}배)")

            results[repo] = {
                "label": label, "key": key, "block": blk,
                "n_rows": int(len(norms)), "median_norm": med,
                "top5": [{"row": i, "norm": n, "ratio": r} for i, n, r in top],
                "max_ratio": ratio, "verdict": verdict,
            }
            norm_vecs[repo] = norms
            if args.cleanup:
                blob = Path(sp).resolve()
                cache_root = blob
                for _ in range(6):
                    cache_root = cache_root.parent
                    if cache_root.name.startswith("models--"):
                        break
                try:
                    blob.unlink()
                    print(f"  샤드 삭제")
                except Exception as e:
                    print(f"  샤드 삭제 실패: {e}")
        except Exception as e:
            print(f"  실패: {type(e).__name__}: {e}")
            results[repo] = {"error": f"{type(e).__name__}: {e}"}

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")

    ok = [(v["label"], v["max_ratio"], v["top5"][0]["row"])
          for v in results.values() if "max_ratio" in v]
    if len(ok) >= 2:
        print("\n" + "=" * 58)
        print(f"{'모델':<28}{'최대 배율':>10}{'행 번호':>10}")
        for lab, r, row in ok:
            print(f"{lab:<28}{r:>9.2f}배{row:>10}")
        # --- 가중치 공유 판별 (해석의 전제) ---
        # Qwen2.5-VL 3B와 7B는 비전 config가 완전히 동일하다(depth 32 / hidden 1280 /
        # intermediate 3420). 같은 ViT에서 초기화됐다면 **같은 행이 튀는 것은 당연**하고
        # "구조적"의 증거가 되지 못한다. 그래서 노름 벡터를 직접 비교한다.
        import itertools
        pairs = [(a, b) for a, b in itertools.combinations(norm_vecs, 2)
                 if norm_vecs[a].shape == norm_vecs[b].shape]
        if pairs:
            print("\n[가중치 공유 판별] 같은 형상 모델 쌍의 행 노름 벡터 비교")
            shared = []
            for a, b in pairs:
                d = float(np.abs(norm_vecs[a] - norm_vecs[b]).max())
                rel = d / float(np.median(norm_vecs[a]))
                same = rel < 1e-3
                shared.append(same)
                print(f"  {a.split('/')[-1]:<28} vs {b.split('/')[-1]:<28} "
                      f"최대 차이 {d:.6f} (중앙값 대비 {rel:.2e})"
                      f"  → {'**동일 가중치**' if same else '다른 가중치'}")
                results.setdefault("_pairwise", []).append(
                    {"a": a, "b": b, "max_abs_diff": d, "rel": rel, "same_weights": same})
            if any(shared):
                print("  ⚠ 동일 가중치 쌍이 있다 — 그 쌍에서 같은 행이 튀는 것은")
                print("    **구조적 필연이 아니라 물려받은 것**이다. 독립 표본만으로 판정할 것.")

        rows = {row for _, _, row in ok}
        if len(rows) == 1:
            print("\n→ 같은 행 번호에 나타난다: **구조적**이다.")
        elif all(r >= 4 for _, r, _ in ok):
            print("\n→ 행 번호는 다르지만 모두 뚜렷한 이상치가 있다: "
                  "**현상은 계열 공통, 위치는 체크포인트마다 다르다.**")
        else:
            print("\n→ 일부 모델에만 나타난다: **이 체크포인트 고유에 가깝다.**")
        print("=" * 58)


if __name__ == "__main__":
    main()
