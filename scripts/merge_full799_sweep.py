"""799장 해상도 스윕 — 모델별 파일을 해상도마다 한 파일로 합친다 (그림 스크립트 입력 형식).

eval_results/full799/res_<max_pixels>__<모델>.json  →  eval_results/full799/resolution_sweep/res_<max_pixels>.json
옛 eval_results/resolution_sweep/res_<max_pixels>.json(모델 전부가 한 파일)과 같은 구조다. `res_1440000_rep2.json`
(재현성 확인 재실행)은 합치지 않는다. 모델마다 _meta(n_images·max_pixels·min_pixels)가 같은지 확인한다.
"""
import json, re
from pathlib import Path
src = Path("eval_results/full799"); out = src / "resolution_sweep"; out.mkdir(exist_ok=True)
groups = {}
for p in sorted(src.glob("res_*__*.json")):
    m = re.fullmatch(r"res_(\d+)__(.+)\.json", p.name)
    if m: groups.setdefault(int(m.group(1)), []).append((m.group(2), p))
for mp, items in sorted(groups.items()):
    merged = {}
    for model, p in items:
        d = json.loads(p.read_text())
        if "_meta" in merged and merged["_meta"] != d["_meta"]:
            raise SystemExit(f"{p.name}: _meta 불일치 {d['_meta']} vs {merged['_meta']}")
        merged["_meta"] = d["_meta"]; merged[model] = d[model]
    (out / f"res_{mp}.json").write_text(json.dumps(merged, ensure_ascii=False))
    print(f"res_{mp}.json  모델 {len(items)}개  n_images={merged['_meta']['n_images']}")
