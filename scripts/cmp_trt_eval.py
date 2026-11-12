"""
비교 처방의 TensorRT 강타입 fp16 엔진 측정 — 개정 15d (2026-11-04, 결과 보기 전 등록).

(R)·무교정·B1·B2·DT-원형 그래프(같은 내보내기 설정: 블록 16 분할, 어텐션 나눠 계산, fp16)를 같은 코드로 잰다:
799장마다 p1→p2 엔진 출력의 비유한 여부(상실), 무교정 PyTorch fp32 원래 구현 출력 대비 코사인(이미지 전체 한 벡터 — 실험 3과 같은 정의 — 과
토큰별 중앙값), 지연(두 엔진 합, 참고 — 최종 지연은 다른 작업이 없을 때 따로). venv_ort(PyTorch + venv_trt의 tensorrt)에서 돈다.

    /workspace/venv_ort/bin/python scripts/cmp_trt_eval.py --cfg resid:A_p1.onnx:A_p2.onnx --cfg B1:B_p1.onnx:B_p2.onnx --out eval_results/cmp/trt_all.json
전처리·기준 출력을 이미지마다 한 번만 계산해 모든 엔진에 넣는다(같은 이미지를 같은 시점에 — 지연 비교도 같은 부하에서).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat                                   # noqa: E402
from dataloader import build_token_to_images       # noqa: E402

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"


def build(onnx_path, plan_path, ws_gb=24):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    if Path(plan_path).exists():
        return rt.deserialize_cuda_engine(Path(plan_path).read_bytes())
    b = trt.Builder(logger)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(net, logger)
    if not p.parse_from_file(str(onnx_path)):
        raise SystemExit("ONNX 파싱 실패: " + "; ".join(str(p.get_error(i)) for i in range(p.num_errors)))
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_gb << 30)
    print(f"[빌드] {Path(onnx_path).name} (강타입)", flush=True)
    plan = b.build_serialized_network(net, cfg)
    if plan is None:
        raise SystemExit("엔진 빌드 실패")
    Path(plan_path).parent.mkdir(parents=True, exist_ok=True)
    Path(plan_path + ".tmp").write_bytes(bytes(plan)); Path(plan_path + ".tmp").replace(plan_path)
    return rt.deserialize_cuda_engine(bytes(plan))


def runner(eng):
    import tensorrt as trt
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    iname = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    oname = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    idt = {trt.float32: torch.float32, trt.float16: torch.float16}[eng.get_tensor_dtype(iname)]
    odt = {trt.float32: torch.float32, trt.float16: torch.float16}[eng.get_tensor_dtype(oname)]
    oshape = tuple(ctx.get_tensor_shape(oname))

    def run(x):
        x = x.to("cuda", idt).contiguous()
        y = torch.empty(oshape, dtype=odt, device="cuda")
        ctx.set_tensor_address(iname, x.data_ptr()); ctx.set_tensor_address(oname, y.data_ptr())
        if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("엔진 실행 실패")
        return y
    return run


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--cfg", action="append", required=True, help="이름:p1.onnx:p2.onnx (여러 번)")
    ap.add_argument("--checkpoint", default="checkpoints/student_baseline_v2/epoch_1")
    ap.add_argument("--max_pixels", type=int, default=1440000)
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--attn", default="eager", help="기준 PyTorch 모델의 어텐션 구현. default = 지정 안 함(공식 사용법, sdpa) — 개정 22 정정 2026-11-04")
    ap.add_argument("--list", default=None, help="이미지 경로 목록 파일(줄마다 하나). 없으면 DriveLM 검증 CAM_FRONT 표준 순서. 개정 23 RealWorldQA — 2026-11-04")
    ap.add_argument("--plan_dir", default="onnx/cmp/plans")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    awq_compat.patch()
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32,
                                                           **({} if a.attn == "default" else {"attn_implementation": a.attn}))
    if a.checkpoint.lower() != "none":
        m = PeftModel.from_pretrained(m, a.checkpoint).merge_and_unload()
    ref_tower = m.visual.eval().to("cuda")                 # 무교정 fp32 원래 구현(기준)
    del m
    cfgs = []
    for c in a.cfg:
        tag, p1, p2 = c.split(":")
        e1 = build(p1, f"{a.plan_dir}/{Path(p1).stem}.plan")
        e2 = build(p2, f"{a.plan_dir}/{Path(p2).stem}.plan")
        cfgs.append((tag, p1, p2, runner(e1), runner(e2)))

    proc = AutoProcessor.from_pretrained(BASE, max_pixels=a.max_pixels, min_pixels=a.min_pixels)
    if a.list:
        paths = [l.strip() for l in open(a.list) if l.strip()][:a.limit]
    else:
        tok = build_token_to_images(VAL)
        paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]
    part = Path(a.out + ".partial.jsonl")
    done = {}
    if part.exists():
        for line in part.read_text().splitlines():
            try:
                r = json.loads(line); done[r["idx"]] = r
            except json.JSONDecodeError:
                continue
    fo = open(part, "a")
    for i, p in enumerate(paths):
        if i in done:
            continue
        enc = proc.image_processor(images=[Image.open(p).convert("RGB")], return_tensors="pt")
        pv, grid = enc["pixel_values"].to("cuda", torch.float32), enc["image_grid_thw"].to("cuda")
        with torch.no_grad():
            ref = ref_tower(pv, grid).float()
        row = {"idx": i, "image": Path(p).name}
        for tag, _p1, _p2, r1, r2 in cfgs:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = r2(r1(pv))
            torch.cuda.synchronize(); lat = (time.perf_counter() - t0) * 1000
            o = out.float()
            fin = bool(torch.isfinite(o).all())
            rr = {"nonfinite": not fin, "latency_ms": lat}
            if fin:
                a_, b_ = ref.double().flatten(), o.double().flatten()
                rr["cos_image"] = float(a_ @ b_ / (a_.norm() * b_.norm()))
                rr["cos_token_median"] = float(torch.nn.functional.cosine_similarity(o, ref, dim=-1).median())
            row[tag] = rr
        fo.write(json.dumps(row) + "\n"); fo.flush()
        done[i] = row
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(paths)} 상실 " + " ".join(f"{t}:{sum(r[t]['nonfinite'] for r in done.values())}" for t, *_ in cfgs), flush=True)
    fo.close()
    rows = [done[i] for i in sorted(done)]
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))] if v else None
    res = {"n": len(rows), "configs": {}}
    for tag, p1, p2, *_ in cfgs:
        rs = [r[tag] for r in rows]
        ci = sorted(x["cos_image"] for x in rs if "cos_image" in x)
        ct = sorted(x["cos_token_median"] for x in rs if "cos_token_median" in x)
        lat = sorted(x["latency_ms"] for x in rs)[2:]
        res["configs"][tag] = {"p1": p1, "p2": p2, "loss": sum(x["nonfinite"] for x in rs),
                               "cos_image": {"min": ci[0] if ci else None, "median": q(ci, 0.5)},
                               "cos_token_median": {"min": ct[0] if ct else None, "median": q(ct, 0.5)},
                               "latency_ms_p50_reference_only": q(lat, 0.5),
                               "engine_mib": {k: Path(f"{a.plan_dir}/{Path(v).stem}.plan").stat().st_size / 2**20 for k, v in (("p1", p1), ("p2", p2))}}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
