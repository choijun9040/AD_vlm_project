"""
기준 ⑤ — 잔차 재조정을 적용한 기준 모델의 원본 해상도 fp16 정확도와, 같은 문항 bf16 출력과의 문자열 일치율. 2026-10-30.
사전 등록: thesis_outline_20260910.md §6 「교정의 전 계열 확장」. 일치율이 Full의 두 형식 간 일치율(92.9%) 이상이면
「정확도 비용 없음」, 미만이면 수치만 보고한다.

    python scripts/compare_fix_predictions.py --fix "eval_results/fixfam/<태그>_shard*of8_predictions.jsonl" \
        --ref "eval_results/full799/student_baseline_v2_epoch_1_native_bf16_shard*of8_predictions.jsonl"
"""
import argparse
import glob
import json
import re


def load(pattern):
    """(조각 번호, 조각 안 문항 번호)를 키로 — (이미지, 질문)을 키로 쓰면 같은 이미지에 같은 질문이 두 번 나오는 84문항이
    합쳐져 11,309가 아니라 11,225문항만 센다(2026-11-01 정정). `idx`는 조각마다 0부터 다시 매겨진다."""
    rows = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"shard(\d+)of\d+", f)
        s = int(m.group(1)) if m else 0
        for ln in open(f):
            r = json.loads(ln)
            rows[(s, r["idx"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="eval_results/fixfam/criterion5.json")
    a = ap.parse_args()
    fx, rf = load(a.fix), load(a.ref)
    keys = sorted(set(fx) & set(rf))
    bad = [k for k in keys if (fx[k]["image"], fx[k]["question"]) != (rf[k]["image"], rf[k]["question"])]
    if bad:
        raise SystemExit(f"두 실행의 문항 순서가 다르다 — {len(bad)}개 (예: {bad[0]})")
    acc_fix = sum(fx[k]["correct"] for k in keys) / len(keys)
    acc_ref = sum(rf[k]["correct"] for k in keys) / len(keys)
    same = sum(fx[k]["pred_raw"] == rf[k]["pred_raw"] for k in keys) / len(keys)
    res = {"n": len(keys), "n_fix": len(fx), "n_ref": len(rf), "acc_fix_fp16": acc_fix, "acc_ref_bf16": acc_ref,
           "diff_pp": (acc_fix - acc_ref) * 100, "string_agreement": same,
           "reference_agreement_full_fp16_vs_bf16": 0.929,
           "verdict": "정확도 비용 없음" if same >= 0.929 else "수치만 보고"}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(a.out, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
