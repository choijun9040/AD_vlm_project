"""
ONNX → TensorRT 엔진을 **빌드만 하고** 파일로 쓴 뒤 끝낸다 (2026-10-02, 사전 등록 6.4h).

trtexec는 `--skipInference`를 줘도 빌드한 엔진을 같은 프로세스에서 다시 적재한다. Orin Nano 8GB에서
802,816 강타입 엔진은 빌드·직렬화까지 끝났는데, 빌드 메모리를 쥔 채 적재하며 가중치 자원 0.64 GB를
할당하지 못해 실패하고 segfault가 났다. 이 스크립트는 직렬화한 바이트를 쓰고 바로 종료한다 — 적재는
측정 스크립트가 새 프로세스에서 한다.

빌더 설정은 TensorRT API 기본값이다(trtexec 기본값과 같다: 최적화 수준 3, 작업 공간 제한 없음).

    python3 orin_build_engine.py --onnx X_p1.onnx --out X_p1_typed.plan --mode typed
    python3 orin_build_engine.py --onnx X_p1.onnx --out X_p1_fp16.plan --mode fp16
"""

import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["typed", "fp16", "fp32"], required=True,
                    help="typed=강타입(--stronglyTyped), fp16=관행 --fp16, fp32=플래그 없음")
    args = ap.parse_args()

    import tensorrt as trt
    logger = trt.Logger(trt.Logger.VERBOSE)          # 표준 오류로 나온다 — 호출 쪽이 로그 파일로 돌린다
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if args.mode == "typed":
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    # 가중치가 외부 데이터(.onnx.data)이므로 parse_from_file을 쓴다(바이트로 넘기면 상대 경로를 못 찾는다)
    if not parser.parse_from_file(str(args.onnx)):
        for i in range(parser.num_errors):
            print("파서 오류:", parser.get_error(i), file=sys.stderr)
        sys.exit(2)

    cfg = builder.create_builder_config()
    if args.mode == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)

    print(f"[빌드] {args.onnx} → {args.out} ({args.mode}, 레이어 {network.num_layers}개, "
          f"TensorRT {trt.__version__})", file=sys.stderr, flush=True)
    t0 = time.time()
    ser = builder.build_serialized_network(network, cfg)
    if ser is None:
        print("[빌드] 실패 — 엔진을 만들지 못했다", file=sys.stderr)
        sys.exit(1)
    Path(args.out).write_bytes(bytes(ser))
    print(f"[빌드] 완료 {time.time() - t0:.0f}초 — {args.out} "
          f"({Path(args.out).stat().st_size / 2**20:.0f} MiB)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
