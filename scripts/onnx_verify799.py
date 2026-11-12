"""
ONNX 래퍼 검증을 799장으로 — 개정 13 (2026-11-04, thesis_outline §6, 결과 보기 전 등록).

`export_vision_tower_onnx.py`는 대표 이미지 1장으로 (a) 래퍼 대 원래 구현 최대 절대오차와 (b) ONNX 런타임 대 PyTorch(래퍼) 출력의
토큰별 코사인 중앙값을 쟀다. 같은 두 값을 DriveLM 검증 CAM_FRONT 799장(표준 순서) 각각에서 잰다. 그래프는 대표 이미지 격자로 형상을
고정하므로 격자가 다른 이미지는 세기만 하고 건너뛴다.

onnxruntime이 공용 환경에 없어 검증 전용 가상환경(`/workspace/venv_ort`, venv_main 패키지를 읽기만 하고 onnxruntime만 따로 설치)에서 돈다.

**--runtime trt (2026-11-04, 개정 13 ① 변경 — 결과 보기 전, 사용자 「A로 진행」).** ONNX 런타임 CPU는 dense 그래프(창 어텐션도 전체 크기 마스크)라
원본 해상도에서 장당 수 분이 걸려 799장 × 4그래프가 며칠이다. 같은 ONNX 그래프를 TensorRT fp32 엔진(TF32 끔)으로 빌드해 실행한다.
venv_ort에 venv_trt 패키지 경로를 읽기 전용으로 붙여 쓴다(tensorrt). 엔진 입출력은 PyTorch 텐서 주소로 넘긴다(cudart 이름 충돌 회피).

    /workspace/venv_ort/bin/python scripts/onnx_verify799.py --checkpoint checkpoints/student_baseline_v2/epoch_1 \
        --max_pixels 1440000 --onnx onnx/verify799/tower_baseline_v2_native.onnx --out eval_results/verify799/onnx_baseline_v2_native.json
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat                                               # noqa: E402
from dataloader import build_token_to_images                    # noqa: E402
from export_vision_tower_onnx import TowerExportWrapper, BASE, DRIVELM_VAL_JSON   # noqa: E402


def trt_runner(onnx_path, plan_path, workspace_gb):
    """같은 ONNX 그래프의 TensorRT fp32 엔진(TF32 끔). 입출력은 PyTorch CUDA 텐서."""
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    rt = trt.Runtime(logger)
    if Path(plan_path).exists():
        eng = rt.deserialize_cuda_engine(Path(plan_path).read_bytes())
    else:
        builder = trt.Builder(logger)
        net = builder.create_network(0)
        parser = trt.OnnxParser(net, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise SystemExit("ONNX 파싱 실패: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
        cfg.clear_flag(trt.BuilderFlag.TF32)                # 진짜 fp32 — 원래 검증(ONNX 런타임 CPU)과 같은 정밀도
        print("[TensorRT] fp32 엔진 빌드 중 …", flush=True)
        plan = builder.build_serialized_network(net, cfg)
        if plan is None:
            raise SystemExit("엔진 빌드 실패")
        Path(plan_path + ".tmp").write_bytes(bytes(plan)); Path(plan_path + ".tmp").replace(plan_path)
        eng = rt.deserialize_cuda_engine(bytes(plan))
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    iname = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    oname = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    oshape = tuple(ctx.get_tensor_shape(oname))

    def run(pv):
        x = pv.to("cuda", torch.float32).contiguous()
        y = torch.empty(oshape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(iname, x.data_ptr()); ctx.set_tensor_address(oname, y.data_ptr())
        if not ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("엔진 실행 실패")
        torch.cuda.synchronize()
        return y
    return run


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_pixels", type=int, required=True)
    ap.add_argument("--min_pixels", type=int, default=50176)    # export_vision_tower_onnx.py 기본값
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--runtime", choices=["ort", "trt"], default="ort")
    ap.add_argument("--plan", default=None, help="--runtime trt: 엔진 파일(있으면 다시 빌드하지 않는다)")
    ap.add_argument("--workspace_gb", type=int, default=40)
    ap.add_argument("--list", default=None, help="이미지 경로 목록(줄마다 하나, 첫 장이 그래프 형상) — 개정 24 RealWorldQA 2026-11-04")
    ap.add_argument("--attn", default="eager", help="원래 구현(비교 기준)의 어텐션 구현. default = 지정 안 함(공식 사용법, sdpa) — 개정 24")
    a = ap.parse_args()

    awq_compat.patch()
    if a.runtime == "ort":
        import onnxruntime as ort
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32, attn_implementation="eager")
    if a.checkpoint.lower() != "none":
        model = PeftModel.from_pretrained(model, a.checkpoint).merge_and_unload()
    tower = model.visual.eval().to("cuda")
    if a.attn == "eager":
        orig = copy.deepcopy(tower)                 # 래퍼가 어텐션 forward를 바꿔 놓으므로 원래 구현은 따로 둔다
    else:                                           # 원래 구현을 공식 사용법(어텐션 구현 지정 안 함)으로 따로 불러온다 — 래퍼는 자체 어텐션을 쓴다
        m2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
        if a.checkpoint.lower() != "none":
            m2 = PeftModel.from_pretrained(m2, a.checkpoint).merge_and_unload()
        orig = m2.visual.eval().to("cuda"); del m2
    del model

    if a.list:
        paths = [l.strip() for l in open(a.list) if l.strip()][:a.limit]
    else:
        tok = build_token_to_images(DRIVELM_VAL_JSON)
        paths = [str(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v][:a.limit]
    proc = AutoProcessor.from_pretrained(BASE, max_pixels=a.max_pixels, min_pixels=a.min_pixels)
    enc0 = proc.image_processor(images=[Image.open(paths[0]).convert("RGB")], return_tensors="pt")
    grid0 = enc0["image_grid_thw"].to("cuda")
    wrapper = TowerExportWrapper(tower, grid0, attn="dense").eval()

    if a.runtime == "ort":
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
        run_graph = lambda pv: torch.from_numpy(sess.run(None, {"pixel_values": pv.cpu().numpy()})[0]).float()
    else:
        run_graph = trt_runner(a.onnx, a.plan or a.out + ".plan", a.workspace_gb)

    part = Path(a.out + ".partial.jsonl")
    done = {}
    if part.exists():
        for line in part.read_text().splitlines():
            try:                                # 인스턴스가 쓰는 도중 꺼지면 마지막 줄이 잘릴 수 있다 — 그 장은 다시 잰다
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[r["idx"]] = r
        part.write_text("".join(json.dumps(done[i]) + "\n" for i in sorted(done)))   # 잘린 줄을 걷어 낸다
    fo = open(part, "a")
    for i, p in enumerate(paths):
        if i in done:
            continue
        enc = proc.image_processor(images=[Image.open(p).convert("RGB")], return_tensors="pt")
        grid = enc["image_grid_thw"]
        row = {"idx": i, "image": Path(p).name, "grid": grid.tolist()}
        if not torch.equal(grid.to("cuda"), grid0):
            row["grid_mismatch"] = True
        else:
            pv = enc["pixel_values"].to("cuda", torch.float32)
            with torch.no_grad():
                ref = orig(pv, grid0)
                got = wrapper(pv)
            row["wrapper_maxabs"] = float((ref.float() - got.float()).abs().max())
            o = run_graph(pv).float().cpu()
            g = got.float().cpu()
            rr = ref.float().cpu()
            fin = torch.isfinite(g).all(dim=-1) & torch.isfinite(o).all(dim=-1) & torch.isfinite(rr).all(dim=-1)
            row["n_tokens"], row["n_nonfinite"] = int(fin.numel()), int((~fin).sum())
            row["ort_cos_median"] = float(torch.nn.functional.cosine_similarity(o[fin], g[fin], dim=-1).median())    # 런타임 대 래퍼(원래 검증과 같은 쌍)
            row["cos_vs_orig_median"] = float(torch.nn.functional.cosine_similarity(o[fin], rr[fin], dim=-1).median())  # 런타임 대 원래 구현
        fo.write(json.dumps(row) + "\n"); fo.flush(); os.fsync(fo.fileno())
        done[i] = row
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)}  최근 코사인 {row.get('ort_cos_median')}  래퍼 오차 {row.get('wrapper_maxabs')}", flush=True)
    fo.close()

    rows = [done[i] for i in sorted(done)]
    ok = [r for r in rows if not r.get("grid_mismatch")]
    cs = sorted(r["ort_cos_median"] for r in ok)
    res = {"checkpoint": a.checkpoint, "max_pixels": a.max_pixels, "onnx": a.onnx, "n_images": len(rows),
           "grid_mismatch": len(rows) - len(ok), "grid": grid0.tolist(),
           "wrapper_maxabs_max": max((r["wrapper_maxabs"] for r in ok), default=None),
           "ort_cos_median": {"min": cs[0] if cs else None, "median": cs[len(cs) // 2] if cs else None},
           "n_below_0.9999": sum(c < 0.9999 for c in cs), "runtime": a.runtime,
           "cos_vs_orig_median": {"min": min((r["cos_vs_orig_median"] for r in ok if "cos_vs_orig_median" in r), default=None)},
           "nonfinite_tokens_total": sum(r["n_nonfinite"] for r in ok)}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    with open(a.out.replace(".json", "_per_image.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
