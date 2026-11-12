"""개정 26 — Qwen 공식 RealWorldQA 평가 코드(QwenLM/Qwen3-VL 커밋 f8dca990, scripts/vendor/qwen_rwqa/ — 고치지 않음)의 프롬프트·해상도·규칙 추출로 공식 Qwen2.5-VL-3B bf16을 평가해
기술 보고서 값(65.4)과 나란히 둔다. 2026-11-04, 결과 보기 전 등록. 바꾼 것은 둘뿐(등록): 추론은 vLLM 대신 transformers(공식 기본 어텐션), 생성은 샘플링 대신 탐욕(최대 2,048토큰);
판정 모델(GPT-3.5)은 쓰지 않는다 — 규칙 추출이 실패하면 공식 코드대로 무작위 선택(시드 0)과, 따로 「실패 = 오답」을 함께 적는다.

    prep  (venv_pq):   python scripts/qwen_rwqa_official.py prep  --lmu /dev/shm/lmu --out /dev/shm/lmu/items.jsonl
    infer (venv_main): python scripts/qwen_rwqa_official.py infer --items /dev/shm/lmu/items.jsonl --out eval_results/cmp/rwqa_official_pred.jsonl
    score (venv_pq):   python scripts/qwen_rwqa_official.py score --lmu /dev/shm/lmu --pred eval_results/cmp/rwqa_official_pred.jsonl --out eval_results/cmp/rwqa_official_score.json
"""
import argparse, json, os, random, sys
from pathlib import Path

V = str(Path(__file__).resolve().parent / "vendor" / "qwen_rwqa")
MIN_PX, MAX_PX = 768 * 28 * 28, 5120 * 28 * 28          # 공식 코드 기본값(run_realworldqa.py 96·97행)


def prep(a):
    sys.path.insert(0, V)
    os.environ["LMUData"] = a.lmu
    from dataset_utils import load_dataset, dump_image, build_realworldqa_prompt
    data = load_dataset("RealWorldQA")                   # 공식 코드가 TSV를 받아 MD5 대조
    img_root = os.path.join(a.lmu, "images", "RealWorldQA")
    rows = []
    for i in range(len(data)):
        line = data.iloc[i]
        msgs = build_realworldqa_prompt(line, lambda l: dump_image(l, img_root), MIN_PX, MAX_PX)
        rows.append({"index": str(line["index"]), "messages": msgs})
    Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"저장: {a.out} ({len(rows)}문항, min_pixels {MIN_PX:,} max_pixels {MAX_PX:,})")


def infer(a):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    base = "Qwen/Qwen2.5-VL-3B-Instruct"
    proc = AutoProcessor.from_pretrained(base)                        # 공식 기본값
    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16).to("cuda").eval()   # 어텐션 지정 안 함
    items = [json.loads(l) for l in open(a.items)]
    done = {}
    if Path(a.out).exists():
        for l in open(a.out):
            r = json.loads(l); done[r["index"]] = r
    fo = open(a.out, "a")
    for k, it in enumerate(items):
        if it["index"] in done:
            continue
        msgs = it["messages"]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img, vid = process_vision_info(msgs, image_patch_size=proc.image_processor.patch_size)   # 공식 코드와 같은 호출
        inp = proc(text=[text], images=img, videos=vid, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = m.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        gen = o[0, inp["input_ids"].shape[1]:]
        pred = proc.batch_decode([gen], skip_special_tokens=True)[0]
        fo.write(json.dumps({"index": it["index"], "gen": pred, "n_tokens": int(gen.numel()),
                             "truncated": int(gen.numel()) >= a.max_new_tokens, "grid": inp["image_grid_thw"].tolist()}, ensure_ascii=False) + "\n"); fo.flush()
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(items)}", flush=True)
    print(f"저장: {a.out}")


def score(a):
    sys.path.insert(0, V)
    os.environ["LMUData"] = a.lmu
    import string
    import pandas as pd
    from dataset_utils import load_dataset
    from eval_utils import eval_single_sample, can_infer, build_choices
    data = load_dataset("RealWorldQA")
    pred = {json.loads(l)["index"]: json.loads(l) for l in open(a.pred)}
    recs = []
    for i in range(len(data)):
        d = data.iloc[i].to_dict(); d.pop("image", None)
        d["prediction"] = str(pred[str(d["index"])]["gen"])
        recs.append(d)
    df = pd.DataFrame.from_records(recs).sort_values(by="index")        # 공식 run_evaluation과 같은 정리
    for k in list(df.keys()):
        df[k.lower() if k not in list(string.ascii_uppercase) else k] = df.pop(k)
    random.seed(0)
    res, rule_fail, strict = [], 0, 0
    for i in range(len(df)):
        item = df.iloc[i].to_dict()
        r = eval_single_sample((None, item))             # 판정 모델 없음 — 규칙 추출, 실패하면 공식 코드대로 무작위
        res.append(r)
        ok = can_infer(item["prediction"], build_choices(item))
        if not ok:
            rule_fail += 1
        strict += int(bool(ok) and ok == item["answer"])
    n = len(res); acc = sum(r["hit"] for r in res) / n
    out = {"n": n, "accuracy_official_code_no_judge_seed0": acc, "accuracy_rule_fail_as_wrong": strict / n, "rule_extract_fail": rule_fail,
           "truncated": sum(int(p["truncated"]) for p in pred.values()), "report_value_qwen25vl_3b": 65.4,
           "per_item": [{k: r[k] for k in ("index", "extracted_answer", "extraction_method", "gt", "hit")} for r in res]}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(f"[공식 코드 채점] 정확도 {acc * 100:.2f}% (무작위 대체 포함, 시드 0) · 실패=오답 {strict / n * 100:.2f}% · 규칙 추출 실패 {rule_fail}/{n} · 잘림 {out['truncated']} · 보고서 65.4")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("stage", choices=["prep", "infer", "score"])
    ap.add_argument("--lmu", default="/dev/shm/lmu")
    ap.add_argument("--items"); ap.add_argument("--pred"); ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    a = ap.parse_args()
    {"prep": prep, "infer": infer, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
