#!/usr/bin/env python3
"""증폭기(출력 투영의 행 노름 이상치)가 Qwen2.5-VL 고유인지 계열을 넘어 있는지 본다.

**왜.** Pixtral 실측(여유 35.91배)으로 *"가변 시퀀스면 붕괴한다"*가 기각됐다
(§7.4). 설계가 원인이 아니라면 남는 것은 **가중치 이상치**다 — Qwen2.5-VL의
`down_proj` 849행은 노름이 중앙값의 9.33배이고 32개 블록 전부에서 최상위다.
**그런 증폭기가 다른 계열에도 있는가**가 원장 L11이다.

**방법.** `probe_weight_distribution.py`와 동일하다 — 출력 투영 가중치의
**행(출력 채널)별 L2 노름**을 구해 `max/p50`을 본다. 순전파가 없으므로
가중치만 내려받으면 되고 GPU도 필요 없다.

출력 투영 이름은 계열마다 다르다: `down_proj`(Qwen·Pixtral),
`fc2`(CLIP·InternVL·Idefics3). 이름 패턴으로 찾고, 못 찾으면 그 사실을 적는다.

실행:
    python scripts/probe_amplifier_across_families.py --repos A B C
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from probe_vision_tower_direct import fetch_vision_state, retry_429  # noqa: E402

# **MLP 출력 투영만 본다 — 우선순위가 필요하다 (2026-09-18).**
# 처음에는 후보를 한 정규식에 넣고 마지막으로 일치한 것을 썼는데, dict 순서에
# 따라 **LLaVA에서 어텐션의 `out_proj`가 MLP의 `fc2`를 덮었다.** 모델마다 다른
# 층을 재면 비교가 무의미하다. 우선순위를 두고 **가장 앞선 것만** 쓴다.
#   down_proj(Qwen·Pixtral) > fc2(CLIP·InternVL·Idefics3) > c_proj
# 어텐션 `out_proj`는 **MLP가 아니므로 후보에서 뺀다.**
OUT_PROJ = re.compile(
    r"(?:^|\.)(?:blocks|layers)\.(\d+)\..*?\b(down_proj|fc2|c_proj)\.weight$")
PRIORITY = {"down_proj": 0, "fc2": 1, "c_proj": 2}


def row_stats(w):
    n = w.float().norm(dim=1)
    s, _ = n.sort(descending=True)
    k = n.numel()
    return {"max": float(s[0]), "p50": float(s[k // 2]),
            "max_over_p50": float(s[0] / s[k // 2]),
            "top_row": int(n.argmax()), "out_features": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--out", default="eval_results/amplifier_across_families.json")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args()

    res = {}
    for repo in args.repos:
        print(f"\n{'='*74}\n[{repo}]\n{'='*74}", flush=True)
        try:
            sd = retry_429(lambda: fetch_vision_state(repo))
            per_block, chosen = {}, {}
            for k, v in sd.items():
                m = OUT_PROJ.search(k)
                if m and v.ndim == 2:
                    bi, pj = int(m.group(1)), m.group(2)
                    if bi in chosen and PRIORITY[chosen[bi]] <= PRIORITY[pj]:
                        continue                      # 더 앞선 후보가 이미 있다
                    chosen[bi] = pj
                    per_block[bi] = row_stats(v)
            names = set(chosen.values())
            proj_name = "/".join(sorted(names)) if names else None
            if len(names) > 1:
                print(f"  ⚠ 블록마다 다른 투영이 잡혔다: {names} — 비교에 쓰지 말 것")
            if not per_block:
                raise RuntimeError(
                    f"출력 투영을 못 찾았다 — 키 예시 {list(sd)[:3]}")
            idx = sorted(per_block)
            last = per_block[idx[-1]]
            ratios = [per_block[i]["max_over_p50"] for i in idx]
            # 같은 행이 전 블록에서 최상위인가 — Qwen2.5-VL의 849 같은 패턴
            tops = [per_block[i]["top_row"] for i in idx]
            dom = max(set(tops), key=tops.count)
            n_dom = tops.count(dom)
            res[repo] = {"proj": proj_name, "n_blocks": len(idx),
                         "last_block": idx[-1], "last": last,
                         "ratio_max": max(ratios), "ratio_p50": sorted(ratios)[len(ratios)//2],
                         "dominant_row": dom, "blocks_where_dominant": n_dom}
            print(f"  {proj_name} · 블록 {len(idx)}개")
            print(f"  마지막 블록  max/p50 = {last['max_over_p50']:.2f}배  "
                  f"(최상위 행 {last['top_row']}/{last['out_features']})")
            print(f"  전 블록      max/p50 최대 {max(ratios):.2f} · 중앙 {sorted(ratios)[len(ratios)//2]:.2f}")
            print(f"  같은 행이 최상위인 블록 수: {n_dom}/{len(idx)} (행 {dom})")
            del sd
        except Exception as e:
            print(f"  실패: {type(e).__name__}: {str(e)[:150]}")
            res[repo] = {"failed": f"{type(e).__name__}: {e}"}
        if args.cleanup:
            import os
            import shutil
            cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            d = Path(cache) / "hub" / ("models--" + repo.replace("/", "--"))
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"  캐시 정리: {d.name}")

    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n{'='*74}")
    print(f"{'모델':40s}{'proj':10s}{'마지막 max/p50':>15s}{'전블록 최대':>12s}{'동일행/블록':>12s}")
    print("-" * 90)
    for r, v in res.items():
        if "failed" in v:
            print(f"{r.split('/')[-1][:38]:40s}{v['failed'][:44]}"); continue
        print(f"{r.split('/')[-1][:38]:40s}{v['proj']:10s}"
              f"{v['last']['max_over_p50']:>14.2f}배{v['ratio_max']:>11.2f}"
              f"{v['blocks_where_dominant']:>7}/{v['n_blocks']}")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
