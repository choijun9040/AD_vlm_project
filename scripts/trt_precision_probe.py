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

# **tensorrt import를 늦춘다 (2026-09-18).** 이 스크립트는 Orin 보드에서 돌지만
# 인자 파싱과 `--help`는 개발기에서도 확인할 수 있어야 한다. 모듈 최상단에서
# import하면 tensorrt 없는 곳에서 `--help`조차 `ModuleNotFoundError`로 죽어,
# 보드에 올리기 전에 플래그 오타를 못 잡는다(O10에서 훑다 발견).
trt = None


def _load_trt():
    """실제로 엔진을 만들 때만 tensorrt를 올린다."""
    global trt
    if trt is None:
        import tensorrt as _trt
        trt = _trt


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
    if mode == "bf16":
        # **가장 명백한 대안.** bf16은 지수부가 fp32와 같아 한계가 3.4e38이므로
        # 이 오버플로가 원리상 일어나지 않는다. 심사에서 "그냥 bf16 쓰면 되지 않나"가
        # 나올 질문이므로 붕괴율과 **지연시간**을 함께 재 답을 준비한다.
        if not hasattr(trt.BuilderFlag, "BF16"):
            raise RuntimeError("이 TensorRT 빌드에 BF16 플래그가 없다")
        cfg.set_flag(trt.BuilderFlag.BF16)
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
    if log_path:
        Path(log_path).write_text("\n".join(lines))   # 실패해도 로그는 남긴다
    if plan is None:
        raise RuntimeError(f"[{mode}] 엔진 빌드 실패 — 로그: {log_path}")
    rt = trt.Runtime(logger)
    return rt.deserialize_cuda_engine(plan), lines


def layer_precisions(lines):
    """빌드 로그의 "Engine Layer Information"에서 텐서 정밀도를 센다.

    주의: 파싱 단계의 `[FLOAT]` 표기는 **ONNX 그래프의 dtype**이지 엔진이 고른
    정밀도가 아니다. 실제 배치는 로그 끝의 Engine Layer Information에
    `Half[...]` / `Float[...]` 형태로 나온다.

    또 하나: TRT는 대부분을 `Myelin` 노드 하나로 융합하므로 **개별 레이어 정밀도가
    보이지 않을 수 있다.** 그때는 Total Weights Memory로 판단한다 — fp16이면
    fp32의 절반이다(실측: 2.7 GB → 1.45 GB).
    """
    prec, weights_mb = {"Half": 0, "Float": 0, "Int8": 0}, None
    in_eli = False
    for l in lines:
        if "Engine Layer Information" in l:
            in_eli = True
            continue
        if "Total Weights Memory" in l:
            m = re.search(r"(\d+)\s*bytes", l)
            if m:
                weights_mb = int(m.group(1)) / 2**20
        if in_eli and l.startswith("Layer("):
            for t in ("Half", "Float", "Int8"):
                prec[t] += len(re.findall(rf"\b{t}\[", l))
    prec["_weights_mb"] = weights_mb
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
    import time
    bad, maxes, lat = 0, [], []
    for x in inputs:
        x = np.ascontiguousarray(x, dtype=i_dt)
        out = np.empty(o_shape, dtype=o_dt)
        t0 = time.perf_counter()
        cudart.cudaMemcpyAsync(d_in, x.ctypes.data, x.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
        ctx.set_tensor_address(i_name, int(d_in))
        ctx.set_tensor_address(o_name, int(d_out))
        ctx.execute_async_v3(stream)
        cudart.cudaMemcpyAsync(out.ctypes.data, d_out, out.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
        cudart.cudaStreamSynchronize(stream)
        lat.append((time.perf_counter() - t0) * 1000)
        if not np.isfinite(out).all():
            bad += 1
        fin = out[np.isfinite(out)]
        if fin.size:
            maxes.append(float(np.abs(fin).max()))
    lat = sorted(lat)[2:]          # 처음 두 번은 워밍업으로 버린다
    return (bad / len(inputs), (float(np.median(maxes)) if maxes else None),
            {"p50": float(np.median(lat)), "p95": float(np.percentile(lat, 95))}
            if lat else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--npz", required=True, help="미리 전처리한 pixel_values 묶음")
    ap.add_argument("--modes", default="fp32,fp16,bf16")
    ap.add_argument("--workspace_gb", type=int, default=24)
    ap.add_argument("--out", default="eval_results/trt_precision_probe.json")
    args = ap.parse_args()
    _load_trt()          # 파싱이 끝난 뒤에 올린다 — --help는 여기 오지 않는다

    z = np.load(args.npz)
    inputs = [z[k] for k in sorted(z.files)]
    print(f"[입력] {len(inputs)}장, 형상 {inputs[0].shape}")
    print(f"[ONNX] {args.onnx}\n")

    res = {"onnx": args.onnx, "n_images": len(inputs), "trt": trt.__version__, "modes": {}}
    for mode in args.modes.split(","):
        mode = mode.strip()
        logp = f"logs/trt_build_{mode}.log"
        try:
            eng, lines = build(args.onnx, mode, args.workspace_gb, logp)
        except Exception as e:
            # 한 조건이 실패해도 나머지는 계속한다. 특히 strict는 TRT 10.12부터
            # layer.precision이 deprecated(강타입으로 대체)라 실패할 수 있는데,
            # **관행 조건이 이미 붕괴를 재현하면 strict는 없어도 결론이 선다.**
            print(f"  [{mode}] 실패 — {type(e).__name__}: {str(e)[:120]}")
            res["modes"][mode] = {"failed": f"{type(e).__name__}: {str(e)[:200]}",
                                  "build_log": logp}
            continue
        prec = layer_precisions(lines)
        nan_rate, absmax, lat = run(eng, inputs)
        n_half, n_float = prec.get("Half", 0), prec.get("Float", 0)
        wmb = prec.get("_weights_mb")
        print(f"  [{mode}] 붕괴 {nan_rate*100:.1f}%   유한 max 중앙값 "
              f"{absmax if absmax is None else f'{absmax:,.0f}'}   "
              f"텐서 Half {n_half} / Float {n_float}   "
              f"가중치 {wmb:,.0f} MiB" if wmb else "" + f"   로그 {logp}")
        if lat:
            print(f"        지연 p50 {lat['p50']:.1f} ms / p95 {lat['p95']:.1f} ms")
        res["modes"][mode] = {"nan_rate": nan_rate, "absmax_p50": absmax,
                              "n_half_tensors": n_half, "n_float_tensors": n_float,
                              "weights_mib": wmb, "latency_ms": lat,
                              "build_log": logp}
        del eng

    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")

    m = res["modes"]
    print("\n" + "=" * 64)
    if "nan_rate" in m.get("fp16", {}) and "nan_rate" in m.get("strict", {}):
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
    elif "nan_rate" in m.get("fp16", {}):
        a = m["fp16"]["nan_rate"]
        if a > 0.5:
            print("판정: **관행 설정(--fp16)에서 붕괴가 재현된다.** 빌더의 레이어별")
            print("  정밀도 선택이 이 오버플로를 막지 못한다. strict 조건은 '관행에서")
            print("  안전할 때'만 필요했으므로 빌드 실패해도 결론에 영향이 없다.")
        else:
            print("판정: 관행 설정에서 안전 — strict가 필요하나 빌드하지 못했다.")
    print("A100 PyTorch 기준값: baseline_v2 원본 해상도 94.8% 붕괴")
    print("=" * 64)


if __name__ == "__main__":
    main()
