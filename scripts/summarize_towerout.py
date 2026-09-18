#!/usr/bin/env python3
"""측정 JSON에서 직접 읽어 블록 뒤 표를 만든다. **손으로 옮겨 적지 않는다.**"""
import json
from pathlib import Path

rows = []

def add(name, blk_hw, tower, post):
    rows.append({"name": name, "blk": blk_hw,
                 "tower_hw": tower["headroom_worst"] if tower else None,
                 "post_hw": post["headroom_worst"] if post else None,
                 "gain": post["gain_vs_block_worst"] if post else
                         (tower["gain_vs_block_worst"] if tower else None)})

for f in ("eval_results/bridge_towerout.json",
          "eval_results/vision_tower_direct_kimi_towerout.json"):
    p = Path(f)
    if not p.exists():
        print(f"  ! {f} 없음"); continue
    for repo, r in json.loads(p.read_text()).items():
        if "failed" in r:
            print(f"  ! {repo}: {r['failed'][:70]}"); continue
        add(repo, r["headroom_worst"], r.get("tower_out"), r.get("post_block"))

# headroom_guard로 잰 둘 (tower_out만 기록, 블록 텐서 제외 개념 없음 —
# Qwen merger는 반환값이 merger 출력 하나뿐이라 그대로 post_block과 같다)
for f, name in (("eval_results/guard_towerout_check.json", "Qwen2.5-VL-3B (baseline_v2)"),
                ("eval_results/guard_qwen2vl_2b_awq_towerout.json", "Qwen2-VL-2B-AWQ")):
    p = Path(f)
    if not p.exists():
        print(f"  ! {f} 없음"); continue
    pb = json.loads(p.read_text())["before"]["per_block"]
    blk = [b for b in pb if b["block"] != "tower_out"][-1]
    to = next((b for b in pb if b["block"] == "tower_out"), None)
    if not to: continue
    t = {"headroom_worst": to["headroom_worst"],
         "gain_vs_block_worst": to["max_worst"] / blk["max_worst"]}
    add(name, blk["headroom_worst"], t, t)

rows.sort(key=lambda r: r["blk"])
print(f"\n{'모델':44s}{'블록':>9s}{'타워반환':>11s}{'블록뒤만':>11s}{'배율':>10s}")
for r in rows:
    f2 = lambda v: f"{v:11.2f}" if v is not None else f"{'—':>11s}"
    g = f"{r['gain']:10.4f}" if r["gain"] is not None else f"{'—':>10s}"
    print(f"{r['name']:44s}{r['blk']:9.2f}{f2(r['tower_hw'])}{f2(r['post_hw'])}{g}")

amp = [r for r in rows if r["gain"] is not None and r["gain"] > 1.05]
print()
if amp:
    print("**키우는 계열이 있다 — 판정이 바뀐다:**", [r["name"] for r in amp])
else:
    mx = max(r["gain"] for r in rows if r["gain"] is not None)
    print(f"모두 줄인다 (최대 배율 {mx:.4f}). **판정 전부 불변.**")
