"""개정 30 — DriveLM 평가·측정 목록을 RealWorldQA 스크립트와 같은 형식(jsonl: idx·image·question·answer)으로 만든다. 2026-11-04.
평가 = DriveLM val CAM_FRONT 799장(표준 순서), 질문 = 이미지마다 첫 NuScenes-QA val 문항. 측정용 = DriveLM train CAM_FRONT 100장(시드 0 — calib_train100.py와 같은 표본).

    python scripts/make_drivelm_official_lists.py --out_dir data/drivelm_official
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataloader import build_token_to_images
from calib_train100 import train_paths

VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
QA = "data/nuscenes_qa/NuScenes_val_questions.json"

ap = argparse.ArgumentParser(allow_abbrev=False); ap.add_argument("--out_dir", required=True); a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
first = {}
for q in json.load(open(QA))["questions"]:
    first.setdefault(q["sample_token"], (q["question"], q.get("answer")))
tok = build_token_to_images(VAL)
rows = []
for t, cams in tok.items():
    if "CAM_FRONT" in cams:
        qq, aa = first.get(t, (None, None))
        rows.append({"idx": len(rows), "image": str(cams["CAM_FRONT"]), "sample_token": t, "question": qq, "answer": aa})
assert len(rows) == 799 and all(r["question"] for r in rows), (len(rows), sum(1 for r in rows if not r["question"]))
(out / "val799.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
paths, pool = train_paths()
(out / "train100.jsonl").write_text("\n".join(json.dumps({"idx": i, "image": p, "question": None, "answer": None}) for i, p in enumerate(paths)) + "\n")
print(f"저장: {out}/val799.jsonl (799), train100.jsonl (100 / {pool})")
