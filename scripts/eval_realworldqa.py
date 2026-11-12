"""개정 22 2단계 — RealWorldQA에서 공식 Qwen2.5-VL-3B의 fp16 상실과 (R) 교정 후 정확도. 2026-11-04, 결과 보기 전 등록(21:21 KST).
765문항을 시드 0으로 섞어 측정용 100(c 결정)·평가용 665. 실행: bf16 무교정 / fp16 무교정 / fp16 (R)(비전 타워만 교정, 모델 전체 fp16 — 기존 정확도 측정과 같은 방식).
채점: 주 지표 lmms-eval(커밋 7bb536b7) realworldqa 채점 — 객관식은 선택지 글자 추출, 그 밖은 완전 일치(2026-11-04 22:27 사용자 허용). 보조 지표로 등록 때의 엄격 완전 일치를 함께. 전처리는 한 번만 해 세 실행이 같이 쓴다.

    /workspace/venv_main/bin/python scripts/eval_realworldqa.py --out eval_results/cmp/rwqa_eval.json
"""
import argparse, copy, json, random, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat; awq_compat.patch()
import qwen_vision_mask_cache; qwen_vision_mask_cache.patch()   # 비전 어텐션 마스크 재사용 — 출력 비트 동일, 속도만(2026-11-10 개정 40)
import headroom_ext as HX
import headroom_guard as G

FP16_MAX, TARGET, SEED, N_CAL = 65504.0, 2.0, 0, 100
PLAN = "eval_results/fixfam/qwen25_pretrained.json"          # 공식 Qwen 원고 (R) 구성 — 자동 도구 구성과 같다(개정 19)


def norm(s):
    s = s.strip().strip(".").strip()
    return s.lower()


sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from lmms_mcq_extract import extract_mcq_answer          # noqa: E402 — lmms-eval 커밋 7bb536b7 그대로


def lmms_score(pred, gt):
    """lmms-eval realworldqa_process_results와 같다 — 객관식(A~D)은 선택지 글자 추출, 그 밖은 lower·strip·끝 마침표 제거 완전 일치."""
    gt = gt.strip()
    if gt.upper() in ("A", "B", "C", "D"):
        return int(extract_mcq_answer(pred, choices=["A", "B", "C", "D"]) == gt.upper())
    return int(pred.lower().strip().rstrip(".") == gt.lower())


def lmms_answer(pred, gt):
    """bf16 대비 답 일치(채점 기준의 답 단위) — 객관식은 추출한 글자, 그 밖은 정규화한 문자열."""
    if gt.strip().upper() in ("A", "B", "C", "D"):
        return extract_mcq_answer(pred, choices=["A", "B", "C", "D"])
    return pred.lower().strip().rstrip(".")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--q", default="data/realworldqa/questions.jsonl")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=12845056,
                    help="공식 처리기 기본값(preprocessor_config.json) — RealWorldQA 이미지(최대 216만 픽셀)를 모두 원본 해상도로. "
                         "2026-11-04 21:28 정정: 처음엔 DriveLM 값 1,440,000을 그대로 써 129/765장이 줄어든 채로 들어갔다")
    ap.add_argument("--calib_q", default=None, help="측정용 목록(jsonl) — 주면 --q 전부가 평가용(개정 30 DriveLM)")
    ap.add_argument("--runs", default="bf16,fp16,fp16_resid", help="개정 27: fp32만 따로 돌릴 때 --runs fp32 --c 0.25")
    ap.add_argument("--c", type=float, default=None, help="주면 측정용 c 결정을 건너뛴다")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_tf32", action="store_true", help="개정 40: 행렬곱·합성곱 TF32 끔(fp32 생성 기준을 순수 fp32로)")
    a = ap.parse_args()
    if a.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    rows = [json.loads(l) for l in open(a.q)]
    if a.calib_q:                                   # 개정 30 DriveLM: 측정용(train 100)을 따로 — 앞에 이어 붙여 앞쪽 = 측정용, 뒤쪽 = 평가용 전부
        crow = [json.loads(l) for l in open(a.calib_q)]
        cal, ev = list(range(len(crow))), list(range(len(crow), len(crow) + len(rows)))
        rows = crow + rows
    else:
        order = list(range(len(rows))); random.Random(SEED).shuffle(order)
        cal, ev = sorted(order[:N_CAL]), sorted(order[N_CAL:])
    proc = AutoProcessor.from_pretrained(G.BASE_3B, max_pixels=a.max_pixels, min_pixels=3136)
    res = {"n_cal": len(cal), "n_eval": len(ev), "seed": SEED, "cal_idx": cal, "max_new_tokens": a.max_new_tokens, "max_pixels": a.max_pixels, "attn_implementation": "기본(지정 안 함 — sdpa)"}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    # c — 측정용 100장, bf16 연산 단위 (--c를 주면 건너뛴다). 끊겨도 이어 하기: 정한 c를 옆 파일에 둔다(2026-11-05)
    cfile = Path(str(a.out) + ".calib.json")
    if a.c is None and cfile.exists():
        res["calib"] = json.loads(cfile.read_text()); a.c = res["calib"]["c"]; print(f"[이어 하기] 측정용 c = {a.c}", flush=True)
    if a.c is not None:
        c = a.c; res.setdefault("calib", {"c": c, "given": True})
    tb = None if a.c is not None else G.load_tower("none", G.BASE_3B, "bfloat16", attn=None)
    cb = G.build_cache(proc, [rows[i]["image"] for i in cal], "bfloat16", device="cpu") if tb is not None else None
    ops = None if tb is None else HX.profile_ops(tb, cb, lambda t, x: G.tower_forward(t, x[0].to("cuda", torch.bfloat16), x[1].to("cuda") if x[1] is not None else None),
                         FP16_MAX, TARGET)
    if ops is not None:
        g = ops["global_max"]; c = HX.choose_c(g["max_worst"], FP16_MAX, TARGET)
        res["calib"] = {"max_worst": g["max_worst"], "point": g["point"], "any_exceed_rate": ops.get("any_exceed_rate"), "c": c}
        print(f"[측정용] 최대 {g['max_worst']:,.0f} @ {g['point']} · 초과 비율 {ops.get('any_exceed_rate')} · c = {c}", flush=True)
        cfile.write_text(json.dumps(res["calib"], default=str))
    del tb, cb, ops; torch.cuda.empty_cache()
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

    # 평가용 전처리 한 번
    inputs = []
    for i in ev:
        r = rows[i]
        msgs = [{"role": "user", "content": [{"type": "image", "image": r["image"]}, {"type": "text", "text": r["question"]}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs.append(proc(text=[text], images=[Image.open(r["image"]).convert("RGB")], return_tensors="pt"))
    print(f"[전처리] 평가용 {len(inputs)}문항", flush=True)

    plan = json.load(open(PLAN))["residual"]
    preds = {}

    def run(name, dtype, fix):
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(G.BASE_3B, torch_dtype=torch.float32 if fix else dtype)   # 어텐션 구현 지정 안 함 — 공식 사용법(기본 sdpa)
        if fix:
            HX.apply_residual(m.visual, lambda t: t.blocks, plan["entry"], c, inplace=True)
            HX.apply_eps(m.visual, plan["eps_norms"], c)
            m = m.to(dtype)
        m = m.to("cuda").eval()
        flag = {}
        h = m.visual.register_forward_hook(lambda _m, _i, o: flag.__setitem__("bad", not torch.isfinite(o if torch.is_tensor(o) else o[0]).all().item()))
        P, B, T = [], [], []
        part = Path(str(a.out) + f".{name}.partial.jsonl")      # 끊겨도 이어 하기 — 문항마다 덧붙인다
        if part.exists():
            for l in part.read_text().splitlines():
                try:
                    r = json.loads(l)
                except json.JSONDecodeError:
                    break
                P.append(r["p"]); B.append(r["b"]); T.append(r["t"])
            print(f"[이어 하기] {name}: {len(P)}문항부터", flush=True)
        fpart = open(part, "a")
        with torch.no_grad():
            for k, inp in enumerate(inputs):
                if k < len(P):
                    continue
                inp = inp.to("cuda"); flag["bad"] = False
                try:
                    o = m.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
                    P.append(proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0])
                except Exception as e:                          # 비유한 입력으로 생성이 죽으면 빈 답으로 센다
                    P.append(""); print(f"  [{name}] {k} 생성 실패: {type(e).__name__}", flush=True)
                B.append(int(flag["bad"]))
                T.append(int(o.shape[1] - inp["input_ids"].shape[1] >= a.max_new_tokens) if P[-1] else 0)
                fpart.write(json.dumps({"p": P[-1], "b": B[-1], "t": T[-1]}, ensure_ascii=False) + "\n"); fpart.flush()
        h.remove(); del m; torch.cuda.empty_cache()
        strict = [int(norm(p) == norm(str(rows[i]["answer"]))) for p, i in zip(P, ev)]
        acc = [lmms_score(p, str(rows[i]["answer"])) for p, i in zip(P, ev)]          # 주 지표 — lmms-eval 채점
        preds[name] = P
        res[name] = {"accuracy": sum(acc) / len(acc), "correct": sum(acc), "accuracy_strict": sum(strict) / len(strict),
                     "vision_nonfinite": sum(B), "truncated": sum(T), "per_q_correct": acc, "per_q_correct_strict": strict, "per_q_bad": B}
        print(f"[{name}] 정확도(lmms-eval) {sum(acc)}/{len(acc)} = {sum(acc) / len(acc):.4f} · 엄격 {sum(strict) / len(strict):.4f} · 비전 비유한 {sum(B)}", flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))

    DT = {"fp32": (torch.float32, False), "bf16": (torch.bfloat16, False), "fp16": (torch.float16, False), "fp16_resid": (torch.float16, True)}
    for r in a.runs.split(","):
        run(r, *DT[r])
    if "fp32" in preds:                                                          # 개정 27·30: 원래 출력(fp32) 대비 답 일치
        bad16 = res.get("fp16", {}).get("per_q_bad", [])
        lost = [k for k, x in enumerate(bad16) if x]
        for name in [n for n in preds if n != "fp32"]:
            s_ = [int(norm(p) == norm(q)) for p, q in zip(preds[name], preds["fp32"])]
            a_ = [int(lmms_answer(p, str(rows[i]["answer"])) == lmms_answer(q, str(rows[i]["answer"]))) for p, q, i in zip(preds[name], preds["fp32"], ev)]
            res[name]["agree_fp32_string"] = sum(s_) / len(s_); res[name]["agree_fp32_answer"] = sum(a_) / len(a_)
            res[name]["lost_items_agree_fp32_string"] = [sum(s_[k] for k in lost), len(lost)]
            print(f"[fp32 대비 일치] {name}: 문자열 {sum(s_) / len(s_):.4f} · 답 {sum(a_) / len(a_):.4f} · 상실 문항 {sum(s_[k] for k in lost)}/{len(lost)}", flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    if "bf16" not in preds or "fp16" not in preds or "fp16_resid" not in preds:   # fp32만 돌린 경우 — 비교는 합치는 단계에서
        res["predictions"] = {n: preds[n] for n in preds}
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str)); print(f"저장: {a.out}"); return
    ref = preds["bf16"]
    for name in ("fp16", "fp16_resid"):
        agree = [int(norm(p) == norm(q)) for p, q in zip(preds[name], ref)]
        res[name]["agree_bf16"] = sum(agree) / len(agree)                        # 출력 문자열 일치(등록한 정의)
        agree_a = [int(lmms_answer(p, str(rows[i]["answer"])) == lmms_answer(q, str(rows[i]["answer"]))) for p, q, i in zip(preds[name], ref, ev)]
        res[name]["agree_bf16_answer"] = sum(agree_a) / len(agree_a)             # 채점 단위의 답 일치(lmms-eval 추출)
    lost = [k for k, b in enumerate(res["fp16"]["per_q_bad"]) if b]
    res["lost_items"] = {"n": len(lost),
                         "fp16_correct": sum(res["fp16"]["per_q_correct"][k] for k in lost),
                         "resid_correct": sum(res["fp16_resid"]["per_q_correct"][k] for k in lost),
                         "bf16_correct": sum(res["bf16"]["per_q_correct"][k] for k in lost),
                         "resid_agree_bf16": sum(int(norm(preds["fp16_resid"][k]) == norm(ref[k])) for k in lost)}
    res["predictions"] = {n: preds[n] for n in preds}
    print(f"[일치] fp16 {res['fp16']['agree_bf16']:.4f} · (R) {res['fp16_resid']['agree_bf16']:.4f} · 상실 문항 {res['lost_items']}", flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
