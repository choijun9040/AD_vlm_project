"""개정 27 — 원래 출력 보존: RealWorldQA 평가용 665문항(시드 0 나누기, 측정용 100으로 c), 비전 인코더 출력을 원래 모델 fp32 출력과 비교. 2026-11-04, 결과 보기 전 등록.
비교: bf16(공식 형식 기준선), fp16 무교정(상실 — 유한한 것만 오차), fp16 (R). 이미지별 fp32 대비 상대 오차(‖차‖/‖기준‖)·코사인.
  qwen  (venv_main):  공식 Qwen2.5-VL-3B, 공식 처리기 기본값, 어텐션 지정 안 함. (R) = 원고 공식 구성(자동 도구와 같음), c는 측정용 100.
  gemma (venv_gemma): 공식 Gemma 4 비전 인코더, max_soft_tokens 280(공식 기본)·1120, 경로 (ii)(풀러·표준화를 모델 형식에서 — 배포 경로), sdpa. (R) = 원고 구성(eps 54 포함), c는 측정용 100.

    python scripts/fidelity_rwqa.py qwen --out eval_results/cmp/fid_qwen.json
    python scripts/fidelity_rwqa.py gemma --model_dir /workspace/gemma4_vision/gemma-4-26B-A4B-it --tokens 280 --out eval_results/cmp/fid_gemma_26B_280.json
"""
import argparse, copy, json, random, sys
from pathlib import Path
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import headroom_ext as HX

FP16_MAX, TARGET, SEED, N_CAL = 65504.0, 2.0, 0, 100
Q = "data/realworldqa/questions.jsonl"


def split(q=Q, calib_q=None):
    rows = [json.loads(l) for l in open(q)]
    if calib_q:                                     # 개정 30 DriveLM: 측정용 목록을 따로(앞쪽 = 측정용, 뒤쪽 = 평가용 전부)
        crow = [json.loads(l) for l in open(calib_q)]
        return crow + rows, list(range(len(crow))), list(range(len(crow), len(crow) + len(rows)))
    o = list(range(len(rows))); random.Random(SEED).shuffle(o)
    return rows, sorted(o[:N_CAL]), sorted(o[N_CAL:])


def stats(ref, outs, per_image=False):
    rel, cos, bad, per = [], [], 0, []
    for y, x in zip(ref, outs):
        x = x.double(); y = y.double()
        if not torch.isfinite(x).all():
            bad += 1; per.append(None); continue
        rel.append(float((x - y).norm() / y.norm()))
        cos.append(float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)))
        per.append([rel[-1], cos[-1]])
    q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else None
    r = {"nonfinite": bad, "n": len(outs), "rel_err": {"median": q(rel, .5), "p99": q(rel, .99), "max": max(rel) if rel else None},
         "cos": {"min": min(cos) if cos else None, "median": q(cos, .5)}}
    if per_image:                                    # 개정 37 ④: 이미지별 [상대 오차, 코사인] — 비유한이면 None
        r["per_image"] = per
    return r


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("which", choices=["qwen", "gemma"])
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--tokens", type=int, default=280)
    ap.add_argument("--q", default=Q)
    ap.add_argument("--calib_q", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_tf32", action="store_true", help="개정 40: 행렬곱·합성곱 TF32 끔(기준 fp32를 순수 fp32로 — Qwen 패치 임베딩은 Conv3d)")
    ap.add_argument("--per_image", action="store_true", help="개정 37 ④: 형식별 이미지 단위 값을 결과에 저장")
    ap.add_argument("--forms", default="bf16,fp16,fp16_resid",
                    help="잴 형식(쉼표). 개정 35: fp32_resid = (R)을 fp32로 — 원래 fp32와의 등가성(799장 전체)")
    a = ap.parse_args()
    forms = a.forms.split(",")
    if "fp32_resid" in forms or a.no_tf32:                            # 등가성은 엄밀한 fp32로 — TF32(행렬곱·합성곱) 끔
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    rows, cal, ev = split(a.q, a.calib_q)
    if a.which == "qwen":
        import awq_compat; awq_compat.patch()
        import qwen_vision_mask_cache; qwen_vision_mask_cache.patch()   # 마스크 재사용 — 출력 비트 동일, 속도만(2026-11-10)
        import headroom_guard as G
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(G.BASE_3B)                          # 공식 기본값
        mk = lambda dt: G.load_tower("none", G.BASE_3B, dt, attn=None)
        prep = lambda idx, dt: G.build_cache(proc, [rows[i]["image"] for i in idx], dt, device="cpu")
        fwd = lambda t, x: G.tower_forward(t, x[0].to("cuda", next(t.parameters()).dtype), x[1].to("cuda"))
        plan = json.load(open("eval_results/fixfam/qwen25_pretrained.json"))["residual"]

        @torch.no_grad()                                 # 가중치 제자리 곱 — 기울기 기록을 끈다(2026-11-05 Gemma에서 실패)
        def fix(t32, c):
            HX.apply_residual(t32, lambda m: m.blocks, plan["entry"], c, inplace=True)
            HX.apply_eps(t32, plan["eps_norms"], c)
            return t32
    else:
        from transformers import AutoImageProcessor, Gemma4Config
        from probe_gemma4_vision import load_vision, forward_points
        ip = AutoImageProcessor.from_pretrained(a.model_dir, max_soft_tokens=a.tokens)

        def mk(dt):
            vm = load_vision(a.model_dir, "float32")
            try:
                vm.config._attn_implementation = "sdpa"                          # 공식 사용법(from_pretrained 기본)과 같게
            except Exception:
                pass
            return vm.to(getattr(torch, dt))

        def prep(idx, dt):
            out = []
            for i in idx:
                e = ip(images=[Image.open(rows[i]["image"]).convert("RGB")], return_tensors="pt")
                out.append((e["pixel_values"], e["image_position_ids"]))
            return out
        fwd = lambda m, x: forward_points(m, x[0], x[1], upcast=False)["final"]

        @torch.no_grad()
        def fix(vm, c):
            vm.patch_embedder.input_proj.weight.mul_(c); vm.patch_embedder.position_embedding_table.mul_(c)
            for lyr in vm.encoder.layers:
                lyr.post_attention_layernorm.weight.mul_(c); lyr.post_feedforward_layernorm.weight.mul_(c)
                lyr.input_layernorm.eps *= c * c; lyr.pre_feedforward_layernorm.eps *= c * c
            vm.std_bias.mul_(c); vm.std_scale.div_(c)
            return vm
    res = {"which": a.which, "model_dir": a.model_dir, "tokens": a.tokens if a.which == "gemma" else None, "n_cal": len(cal), "n_eval": len(ev)}
    import resume as RS
    RS.setup(a.out)                                      # 끊겨도 이어 하기 — c와 형식별 결과를 저장, 남은 형식만 다시(fp32 기준은 다시 계산)
    res.update(RS.load("res") or {})
    # c — 측정용 100, bf16
    if "calib" not in res:
        tb = mk("bfloat16"); cb = prep(cal, "bfloat16")
        with torch.no_grad():
            ops = HX.profile_ops(tb, cb, fwd, FP16_MAX, TARGET)
        g = ops["global_max"]
        res["calib"] = {"max_worst": g["max_worst"], "point": g["point"], "c": HX.choose_c(g["max_worst"], FP16_MAX, TARGET)}
        del tb, cb, ops; torch.cuda.empty_cache(); RS.save("res", res)
    c = res["calib"]["c"]
    print(f"[측정용] 최대 {res['calib']['max_worst']:,.0f} @ {res['calib']['point']} · c = {c}", flush=True)
    # 평가용 — 같은 입력을 형식마다
    todo = [f for f in forms if f not in res]
    x32 = prep(ev, "float32") if todo else []
    outs = {}
    for name, dt, do_fix in (("fp32", "float32", False), ("bf16", "bfloat16", False), ("fp16", "float16", False), ("fp16_resid", "float16", True),
                             ("fp32_resid", "float32", True)):
        if name != "fp32" and name not in todo:
            continue
        if name == "fp32" and not todo:
            continue
        if do_fix:
            m = fix(mk("float32"), c).to(getattr(torch, dt))
        else:
            m = mk(dt)
        with torch.no_grad():
            outs[name] = [fwd(m, (x[0], x[1])).float().cpu() for x in x32]
        del m; torch.cuda.empty_cache()
        if name != "fp32":
            res[name] = stats(outs["fp32"], outs[name], a.per_image)
            outs.pop(name)
            RS.save("res", res)
        Path(a.out + ".partial").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    for name in forms:
        print(f"[{name}] {json.dumps({k: v for k, v in res[name].items() if k != 'per_image'}, ensure_ascii=False)}", flush=True)
    if a.per_image and "fp16" in res and "fp16_resid" in res:           # 개정 37 ④: 무교정 fp16이 유한한 이미지에서 같은 표본 비교
        keep = [i for i, v in enumerate(res["fp16"]["per_image"]) if v is not None]
        q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else None
        for name in ("fp16", "fp16_resid", "bf16"):
            if name in res:
                rr = [res[name]["per_image"][i] for i in keep if res[name]["per_image"][i] is not None]
                res[f"same_subset_{name}"] = {"n": len(rr), "rel_err": {"median": q([r[0] for r in rr], .5), "p99": q([r[0] for r in rr], .99)},
                                              "cos": {"min": min(r[1] for r in rr), "median": q([r[1] for r in rr], .5)}}
                print(f"[같은 {len(keep)}장 · {name}] {json.dumps(res[f'same_subset_{name}'])}", flush=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))   # 다 끝난 뒤에만 결과 파일을 쓴다
    RS.done(); Path(a.out + ".partial").unlink(missing_ok=True)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
