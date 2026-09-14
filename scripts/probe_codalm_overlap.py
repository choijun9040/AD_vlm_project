"""
CODA-LM 이미지가 DriveLM 학습 장면과 겹치는지 **이미지 자체로** 확인한다.

CODA는 KITTI(309) / nuScenes(134) / ONCE(1,057) 장면에서 구성됐으므로 약 9%가
nuScenes 유래다. 그 장면들이 우리가 학습에 쓴 696개 DriveLM 장면과 겹치는지가
"largely domain-disjoint"라는 서술의 근거였고, 지금까지 확인되지 않았다.

**메타데이터로는 확인할 수 없다** — CODA-LM llava-format의 id는 `Mini_general_0`
같은 일련번호뿐이라 원출처가 남아 있지 않다. 그래서 이미지 자체를 대조한다.
겹친다면 같은 사진이므로, 크기가 달라도 지각 해시가 잡아낸다.

방법: dHash(64bit) + aHash(64bit)로 1차 후보를 추리고, 후보에 대해서만
32x32 그레이스케일 정규화 상관을 계산해 확정한다. CODA-LM 이미지는 ~720p로
재인코딩돼 있으므로 완전 일치는 기대할 수 없고, 해밍거리 임계를 둔다.

실행:
    python scripts/check_codalm_overlap.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent))
from dataloader import build_token_to_images

TRAIN_JSON = "data/QA_dataset_nus/v1_0_train_nus.json"


def dhash(img, size=8):
    a = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def ahash(img, size=8):
    a = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    return np.packbits((a > a.mean()).ravel())


def thumb(img, size=32):
    a = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    a -= a.mean()
    n = np.linalg.norm(a)
    return (a / n).ravel() if n > 0 else a.ravel()


POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codalm_dir", default="data/codalm_mini")
    ap.add_argument("--hamming", type=int, default=12,
                    help="후보로 볼 최대 해밍거리(64bit 중). 재인코딩·리사이즈 여유")
    ap.add_argument("--corr", type=float, default=0.92,
                    help="확정 판정 상관계수 임계")
    ap.add_argument("--out", default="eval_results/codalm_overlap.json")
    args = ap.parse_args()

    coda = sorted(Path(args.codalm_dir).glob("*.jpg"))
    print(f"[CODA-LM] {len(coda)}장")

    tok = build_token_to_images(TRAIN_JSON)
    train = sorted({str(p) for v in tok.values() for p in v.values()})
    train = [p for p in train if Path(p).exists()]
    print(f"[DriveLM 학습] 전 카메라 {len(train)}장 (CAM_FRONT 외 5개 포함)")

    print("[해시] CODA-LM …", flush=True)
    c_d, c_a, c_t = [], [], []
    for p in coda:
        im = Image.open(p)
        c_d.append(dhash(im)); c_a.append(ahash(im)); c_t.append(thumb(im))
    c_d, c_a, c_t = np.array(c_d), np.array(c_a), np.array(c_t)

    print("[해시] 학습 이미지 …", flush=True)
    t_d, t_a = [], []
    for i, p in enumerate(train):
        im = Image.open(p)
        t_d.append(dhash(im)); t_a.append(ahash(im))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(train)}", flush=True)
    t_d, t_a = np.array(t_d), np.array(t_a)

    print("[대조] 해밍거리 계산 …", flush=True)
    hits, best_rows = [], []
    for i in range(len(coda)):
        hd = POP[np.bitwise_xor(t_d, c_d[i])].sum(axis=1)
        ha = POP[np.bitwise_xor(t_a, c_a[i])].sum(axis=1)
        comb = np.minimum(hd, ha)
        j = int(comb.argmin())
        best_rows.append({"coda": coda[i].name, "best_train": Path(train[j]).name,
                          "hamming_d": int(hd[j]), "hamming_a": int(ha[j])})
        cand = np.where(comb <= args.hamming)[0]
        # 해시가 가까운 후보 전부의 상관을 재고 **최고값을 기록**한다.
        # 해시 충돌만으로 "겹친다/안 겹친다"를 말할 수 없기 때문이다 — aHash는
        # 도로 장면처럼 밝기 분포가 비슷하면 거리 0까지 충돌한다.
        best_corr, best_j = -1.0, None
        for j2 in cand:
            corr = float(np.dot(c_t[i], thumb(Image.open(train[j2]))))
            if corr > best_corr:
                best_corr, best_j = corr, int(j2)
            if corr >= args.corr:
                hits.append({"coda": coda[i].name, "train": Path(train[j2]).name,
                             "hamming": int(comb[j2]), "corr": round(corr, 4)})
        best_rows[-1]["n_candidates"] = int(len(cand))
        best_rows[-1]["best_corr"] = round(best_corr, 4) if best_j is not None else None
        best_rows[-1]["best_corr_train"] = Path(train[best_j]).name if best_j is not None else None

    dists = np.array([min(r["hamming_d"], r["hamming_a"]) for r in best_rows])
    print("\n" + "=" * 62)
    print(f"확정 중복: {len(hits)}건")
    print(f"최근접 해밍거리 분포: 최소 {dists.min()} / 중앙값 {int(np.median(dists))} / "
          f"최대 {dists.max()}  (임계 {args.hamming})")
    if hits:
        print("\n중복 예시:")
        for h in hits[:10]:
            print(f"  {h['coda']}  ↔  {h['train']}  (해밍 {h['hamming']}, 상관 {h['corr']})")
    else:
        print("\n판정: CODA-LM Mini 143장 중 DriveLM 학습 이미지와 겹치는 것이 **없다**.")
        print("      → '학습 장면과 분리되어 있다'를 실측으로 뒷받침할 수 있다.")
    print("=" * 62)

    corrs = np.array([r["best_corr"] for r in best_rows if r.get("best_corr") is not None])
    top = sorted([r for r in best_rows if r.get("best_corr") is not None],
                 key=lambda r: -r["best_corr"])[:10]
    if corrs.size:
        print(f"\n후보 중 최고 상관 분포: 최대 {corrs.max():.4f} / "
              f"중앙값 {np.median(corrs):.4f} / 최소 {corrs.min():.4f}  (임계 {args.corr})")
        print("상관 상위 5쌍 — 해시가 가까웠던 것들이 실제로 얼마나 닮았는가:")
        for r in top[:5]:
            print(f"  corr={r['best_corr']:.4f}  해밍 d={r['hamming_d']}/a={r['hamming_a']}  "
                  f"{r['coda'][:30]} ↔ {str(r['best_corr_train'])[:46]}")

    res = {"n_codalm": len(coda), "n_train_images": len(train),
           "hamming_threshold": args.hamming, "corr_threshold": args.corr,
           "n_overlap": len(hits), "overlaps": hits[:50],
           "nearest_hamming": {"min": int(dists.min()),
                               "p50": int(np.median(dists)),
                               "max": int(dists.max())},
           "best_corr": ({"max": float(corrs.max()), "p50": float(np.median(corrs)),
                          "min": float(corrs.min())} if corrs.size else None),
           "top_corr_pairs": top,
           "nearest_examples": best_rows[:10]}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
