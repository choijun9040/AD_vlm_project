"""TensorRT가 fp16 빌드에서 **오버플로 레이어를 자동으로 fp32로 남기는가** (7장 7.2, 약점 F).

**왜 필수인가.** 본 논문은 "fp16 배포에서 비전 타워 마지막 블록이 표현 범위를 넘어
붕괴한다"고 주장한다. 그런데 `trtexec --fp16`은 fp16을 *허용*하는 것이지 강제가 아니며,
TensorRT는 정확도를 위해 레이어별로 fp32를 남길 수 있다. **자동으로 회피된다면
"아무도 재지 않아 위험하다"는 문제 제기 자체가 약해진다.** 심사에서 나올 질문이고
지금 답이 없다.

이것은 **Orin에 대한 질문이 아니라 TensorRT의 정밀도 배치 정책에 대한 질문**이므로
A100에서 답할 수 있다. 보드 작업보다 먼저 해야 한다 — 결과에 따라 7장에서 무엇을
재야 하는지가 바뀐다.

세 조건을 빌드해 (a) 마지막 블록 레이어들이 실제로 어떤 정밀도로 배치됐는지,
(b) 같은 250장에서 출력에 NaN/Inf가 나오는 비율을 본다.

  관행   : fp16 허용 (실제 배포 관행)
  엄격   : fp16 강제 (OBEY precision constraints + TF32 끔)
  대조군 : fp32

실행:
    /venv/trt/bin/python scripts/trt_precision_probe.py \
        --onnx onnx/tower_baseline_v2_native.onnx --npz trt_inputs.npz
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import tensorrt as trt


def build(onnx_path, mode, workspace_gb=24, log_path=None):
    """엔진을 빌드하고 (engine, 빌드 로그) 를 돌려준다."""
    lines = []

    class Cap(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
        def log(self, severity, msg):
            lines.append(f"[{severity.name}] {msg}")

    logger = Cap()
    logger.min_severity = trt.ILogger.Severity.VERBOSE
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    # **parse(bytes)가 아니라 parse_from_file을 써야 한다.** 이 그래프는 가중치를
    # 외부 데이터(.onnx.data)로 두는데, 바이트만 넘기면 파서가 그 상대 경로를
    # 해석하지 못해 "Failed to import initializer"로 죽는다(실측).
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print("  파서 오류:", parser.get_error(i))
        raise SystemExit("ONNX 파싱 실패")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if mode in ("fp16", "strict"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    if mode == "strict":
        # fp16을 **강제**한다: 정밀도 제약을 그대로 지키고 TF32도 끈다
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        if hasattr(trt.BuilderFlag, "TF32"):
            cfg.clear_flag(trt.BuilderFlag.TF32)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type not in (trt.LayerType.SHAPE, trt.LayerType.CONSTANT,
                                  trt.LayerType.IDENTITY):
                layer.precision = trt.float16
    print(f"  [{mode}] 빌드 중 (레이어 {network.num_layers}개) …", flush=True)
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise SystemExit(f"[{mode}] 엔진 빌드 실패")
    if log_path:
        Path(log_path).write_text("\n".join(lines))
    rt = trt.Runtime(logger)
    return rt.deserialize_cuda_engine(plan), lines


def layer_precisions(lines):
    """빌드 로그에서 레이어별 배치 정밀도를 긁는다."""
    pat = re.compile(r"(\S+).*?Tactic.*?(Float|Half|Int8)", re.I)
    prec = {}
    for l in lines:
        if "Layer(" in l or "running on" in l.lower() or "Tactic" in l:
            m = re.search(r"(Half|Float|Int8)", l)
            if m:
                name = l.split("]")[-1].strip()[:90]
                prec.setdefault(m.group(1), []).append(name)
    return prec


def run(engine, inputs):
    # cuda-python 13.x부터 경로가 cuda.bindings.runtime으로 바뀌었다(구버전은 cuda.cudart).
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart
    ctx = engine.create_execution_context()
    i_name, o_name = engine.get_tensor_name(0), engine.get_tensor_name(1)
    i_dt = trt.nptype(engine.get_tensor_dtype(i_name))
    o_dt = trt.nptype(engine.get_tensor_dtype(o_name))
    o_shape = tuple(engine.get_tensor_shape(o_name))
    nb_i = int(np.prod(tuple(engine.get_tensor_shape(i_name)))) * np.dtype(i_dt).itemsize
    nb_o = int(np.prod(o_shape)) * np.dtype(o_dt).itemsize
    d_in = cudart.cudaMalloc(nb_i)[1]
    d_out = cudart.cudaMalloc(nb_o)[1]
    stream = cudart.cudaStreamCreate()[1]
    bad, maxes = 0, []
    for x in inputs:
        x = np.ascontiguousarray(x, dtype=i_dt)
        out = np.empty(o_shape, dtype=o_dt)
        cudart.cudaMemcpyAsync(d_in, x.ctypes.data, x.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        ctx.set_tensor_address(i_name, int(d_in))
        ctx.set_tensor_address(o_name, int(d_out))
        ctx.execute_async_v3(stream)
        cudart.cudaMemcpyAsync(out.ctypes.data, d_out, out.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)
        if not np.isfinite(out).all():
            bad += 1
        fin = out[np.isfinite(out)]
        if fin.size:
            maxes.append(float(np.abs(fin).max()))
    return bad / len(inputs), (float(np.median(maxes)) if maxes else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--npz", required=True, help="미리 전처리한 pixel_values 묶음")
    ap.add_argument("--modes", default="fp32,fp16,strict")
    ap.add_argument("--workspace_gb", type=int, default=24)
    ap.add_argument("--out", default="eval_results/trt_precision_probe.json")
    args = ap.parse_args()

    z = np.load(args.npz)
    inputs = [z[k] for k in sorted(z.files)]
    print(f"[입력] {len(inputs)}장, 형상 {inputs[0].shape}")
    print(f"[ONNX] {args.onnx}\n")

    res = {"onnx": args.onnx, "n_images": len(inputs), "trt": trt.__version__, "modes": {}}
    for mode in args.modes.split(","):
        mode = mode.strip()
        logp = f"logs/trt_build_{mode}.log"
        eng, lines = build(args.onnx, mode, args.workspace_gb, logp)
        prec = layer_precisions(lines)
        nan_rate, absmax = run(eng, inputs)
        n_half = len(prec.get("Half", [])); n_float = len(prec.get("Float", []))
        print(f"  [{mode}] 붕괴 {nan_rate*100:.1f}%   유한 max 중앙값 "
              f"{absmax if absmax is None else f'{absmax:,.0f}'}   "
              f"레이어 Half {n_half} / Float {n_float}   로그 {logp}")
        res["modes"][mode] = {"nan_rate": nan_rate, "absmax_p50": absmax,
                              "n_half": n_half, "n_float": n_float,
                              "build_log": logp}
        del eng

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")

    m = res["modes"]
    print("\n" + "=" * 64)
    if "fp16" in m and "strict" in m:
        a, b = m["fp16"]["nan_rate"], m["strict"]["nan_rate"]
        if b > 0.5 and a < 0.1:
            print("판정: **엄격에서만 붕괴** — TensorRT 기본 정책이 오버플로를 회피한다.")
            print("  → 주장을 축소하지 말고 **조건을 특정**할 것:")
            print("     '순수 fp16 경로에서 발생하며, TensorRT 기본 정책은 이를 우연히 회피한다.'")
            print("     같은 가중치가 런타임 설정에 따라 갈리므로 **여유 측정의 필요성은 커진다.**")
        elif a > 0.5:
            print("판정: **관행 설정에서도 붕괴** — 주장이 실제 배포 도구에서 성립한다.")
        else:
            print("판정: **둘 다 안전** — fp16 경로가 만들어지지 않았을 수 있다.")
            print("  빌드 로그에서 마지막 블록 레이어의 정밀도를 직접 확인할 것.")
    print("A100 PyTorch 기준값: baseline_v2 원본 해상도 94.8% 붕괴")
    print("=" * 64)


if __name__ == "__main__":
    main()
