"""개정 22 — RealWorldQA(xai-org/RealworldQA, 765문항)를 받아 이미지 파일과 문항 jsonl로 푼다. venv_pq(pyarrow 격리 환경)에서 돈다. 2026-11-04.

    /workspace/venv_pq/bin/python scripts/fetch_realworldqa.py --out data/realworldqa
"""
import argparse, io, json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(2):
        f = hf_hub_download("xai-org/RealworldQA", f"data/test-0000{k}-of-00002.parquet", repo_type="dataset")
        t = pq.read_table(f).to_pylist()
        for r in t:
            i = len(rows)
            img = r["image"]
            raw = img["bytes"] if isinstance(img, dict) else img
            im = Image.open(io.BytesIO(raw))
            ext = (im.format or "PNG").lower().replace("jpeg", "jpg")
            p = out / "images" / f"{i:04d}.{ext}"
            p.write_bytes(raw)                                # 원본 바이트 그대로(재부호화하지 않는다)
            rows.append({"idx": i, "image": str(p), "width": im.size[0], "height": im.size[1], "question": r["question"], "answer": r["answer"]})
    (out / "questions.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    ws = sorted(r["width"] * r["height"] for r in rows)
    print(f"저장: {out} ({len(rows)}문항) · 픽셀 수 최소 {ws[0]:,} 중앙 {ws[len(ws)//2]:,} 최대 {ws[-1]:,}")


if __name__ == "__main__":
    main()
