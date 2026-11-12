"""개정 27 — 정확도 2단계(bf16·fp16·(R))와 fp32 생성을 합쳐 원래 출력(fp32) 대비 답 일치율을 낸다. 2026-11-04.
    python scripts/merge_rwqa_fp32.py --main eval_results/cmp/rwqa_eval_official.json --fp32 eval_results/cmp/rwqa_eval_fp32.json --out eval_results/cmp/rwqa_agree_fp32.json
"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_realworldqa import norm, lmms_answer, SEED, N_CAL

ap = argparse.ArgumentParser(allow_abbrev=False)
ap.add_argument("--main", required=True); ap.add_argument("--fp32", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open("data/realworldqa/questions.jsonl")]
o = list(range(len(rows))); random.Random(SEED).shuffle(o); ev = sorted(o[N_CAL:])
M = json.load(open(a.main)); F = json.load(open(a.fp32))
ref = F["predictions"]["fp32"]
out = {"n": len(ev), "fp32_accuracy_lmms": F["fp32"]["accuracy"]}
bad = M["fp16"]["per_q_bad"]
for name in ("bf16", "fp16", "fp16_resid"):
    P = M["predictions"][name]
    s_ = [int(norm(p) == norm(q)) for p, q in zip(P, ref)]
    a_ = [int(lmms_answer(p, rows[i]["answer"]) == lmms_answer(q, rows[i]["answer"])) for p, q, i in zip(P, ref, ev)]
    lost = [k for k, b in enumerate(bad) if b]
    out[name] = {"agree_fp32_string": sum(s_) / len(s_), "agree_fp32_answer": sum(a_) / len(a_),
                 "lost_items_agree_fp32_answer": (sum(a_[k] for k in lost), len(lost))}
    print(name, json.dumps(out[name], ensure_ascii=False))
Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
