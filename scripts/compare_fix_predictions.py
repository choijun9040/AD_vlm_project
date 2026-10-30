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


def load(pattern):
    rows = {}
    for f in sorted(glob.glob(pattern)):
        for ln in open(f):
            r = json.loads(ln)
            rows[(r["image"], r["question"])] = r
    return rows


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--fix", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="eval_results/fixfam/criterion5.json")
    a = ap.parse_args()
    fx, rf = load(a.fix), load(a.ref)
    keys = sorted(set(fx) & set(rf))
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
