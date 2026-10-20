"""
Gemma 4 비전 인코더만 내려받기 — 범위 확장 A (사전 등록 thesis_outline §6 「범위 확장 A·B」).

비전 인코더(1.139 GB)가 49.9 GB 샤드 안에 있어 통째로 받지 않고, safetensors 헤더를 읽어
`model.vision_tower.*`·`model.embed_vision.*` 텐서만 범위 요청으로 받아 별도 safetensors로 저장한다.
두 체크포인트(31B·26B-A4B)의 비전 인코더 355개 텐서를 전부 해시로 대조해 같은지도 확인한다.

    python scripts/fetch_gemma4_vision.py --out_dir /workspace/gemma4_vision
"""
import argparse, hashlib, json, struct, urllib.request
from pathlib import Path

REPOS = ["google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it"]
PREFIXES = ("model.vision_tower.", "model.embed_vision.")
SMALL = ["config.json", "processor_config.json", "preprocessor_config.json", "tokenizer_config.json",
         "tokenizer.json", "special_tokens_map.json", "chat_template.jinja", "chat_template.json",
         "generation_config.json"]


def get(url, a=None, b=None):
    h = {"Range": f"bytes={a}-{b}"} if a is not None else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300).read()


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    hashes = {}
    for repo in REPOS:
        base = f"https://huggingface.co/{repo}/resolve/main/"
        out = Path(args.out_dir) / repo.split("/")[1]; out.mkdir(parents=True, exist_ok=True)
        for fn in SMALL:
            try: (out / fn).write_bytes(get(base + fn))
            except Exception: pass
        idx = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
        shards = sorted({v for k, v in idx.items() if k.startswith(PREFIXES)})
        header_out, blobs, off = {}, [], 0
        hashes[repo] = {}
        for sh in shards:
            n = struct.unpack("<Q", get(base + sh, 0, 7))[0]
            hd = json.loads(get(base + sh, 8, 7 + n))
            keys = sorted(k for k in hd if k.startswith(PREFIXES))
            # **접두사별로** 연속 구간을 받는다. 한 구간으로 묶으면 키 순 저장 때문에 두 접두사 사이의
            # 언어 모델 텐서(약 49 GB)까지 받게 된다 — 2026-10-20 실제로 그렇게 멈췄다.
            raws = {}
            for pre in PREFIXES:
                ks = [k for k in keys if k.startswith(pre)]
                if not ks: continue
                lo = min(hd[k]["data_offsets"][0] for k in ks); hi = max(hd[k]["data_offsets"][1] for k in ks)
                span = hi - lo
                assert span < 1.5 * sum(hd[k]["data_offsets"][1] - hd[k]["data_offsets"][0] for k in ks), \
                    f"{pre} 구간이 연속이 아니다 ({span/1e9:.2f} GB)"
                raws[pre] = (lo, get(base + sh, 8 + n + lo, 8 + n + hi - 1))
            for k in keys:
                lo, raw = raws[next(p for p in PREFIXES if k.startswith(p))]
                a, b = hd[k]["data_offsets"]; buf = raw[a - lo:b - lo]
                header_out[k] = {"dtype": hd[k]["dtype"], "shape": hd[k]["shape"], "data_offsets": [off, off + len(buf)]}
                blobs.append(buf); off += len(buf)
                hashes[repo][k] = hashlib.sha256(buf).hexdigest()
        hb = json.dumps(header_out).encode(); hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(out / "vision.safetensors", "wb") as f:
            f.write(struct.pack("<Q", len(hb))); f.write(hb)
            for b in blobs: f.write(b)
        print(f"{repo}: 텐서 {len(header_out)}개, {off/1e9:.3f} GB → {out/'vision.safetensors'}")
    a, b = (hashes[r] for r in REPOS)
    same = sum(1 for k in a if a[k] == b.get(k)); vt = [k for k in a if k.startswith(PREFIXES[0])]
    same_vt = sum(1 for k in vt if a[k] == b.get(k))
    print(f"[대조] 비전 인코더 {same_vt}/{len(vt)}개 같음 · embed_vision 포함 전체 {same}/{len(a)}개 같음")
    json.dump({"same_vision_tower": same_vt == len(vt), "n": len(vt), "same": same_vt},
              open(Path(args.out_dir) / "compare.json", "w"))


if __name__ == "__main__":
    main()
