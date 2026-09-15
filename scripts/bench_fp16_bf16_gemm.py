"""fp16 대 bf16 순수 GEMM 마이크로벤치 — TensorRT bf16 1.41배의 원인 분리.

**왜 이 실험인가.** §7.2에서 bf16 엔진이 fp16보다 1.41배 느렸고, 이를 "형식 선택
자체가 교환"이라고 해석해 문서에 적었다. 그런데 A100(SM80)은 **규격상 fp16과 bf16의
텐서코어 처리량이 같다**(둘 다 312 TFLOPS dense). 그렇다면 1.41배는 형식의 산술
비용일 수 없고, 런타임(TensorRT Myelin)의 bf16 커널 커버리지 문제여야 한다.

빌드 로그가 이미 그 방향을 가리킨다 — fp16은 tactic/autotuning 줄이 707개인데
bf16은 265개뿐이고(2.7배 적다), fp16만 `Half(...1:2...)`, `Half(...1:8...)` 같은
벡터화 레이아웃을 시도하며, 활성 메모리는 bf16이 9% 더 크다(320 대 293 MB).

이 스크립트는 **런타임을 빼고** 같은 형상의 행렬곱만 잰다. 두 형식이 같게 나오면
1.41배의 원인이 형식이 아님이 확정된다.

**사전 등록 (결과 보기 전 작성, 2026-09-15).**
  - 두 형식이 같다(±5% 이내) → §7.2의 "형식 선택 자체가 교환"을 **철회**하고
    "현재 런타임의 bf16 커널 성숙도 비용"으로 고친다. 커널이 성숙하면 bf16이
    사실상 공짜가 되므로 **처방의 수명이 짧다는 한계를 9장에 명시**한다.
    논문에 불리한 방향이지만 그대로 쓴다.
  - bf16이 실제로 유의하게 느리다 → 현재 서술을 유지한다.
  - fp16이 오히려 느리다 → 측정 오류를 먼저 의심한다(워밍업·클럭·전력 제한).

형상은 Qwen2.5-VL-3B 비전 타워 실제 값이다 — hidden 1280, intermediate 3420,
heads 16, 원본 해상도 시퀀스 7,296(= TRT 로그의 pixel_values[7296,1176]).
"""
import argparse
import json
import time

import torch

S, H, I, NH = 7296, 1280, 3420, 16
HD = H // NH

# (이름, M, K, N) — 타워 한 블록이 실제로 도는 GEMM
SHAPES = [
    ("attn.qkv",      S, H, 3 * H),
    ("attn.proj",     S, H, H),
    ("mlp.gate_proj", S, H, I),
    ("mlp.up_proj",   S, H, I),
    ("mlp.down_proj", S, I, H),
]
# 어텐션 배치 GEMM (윈도우 없이 전체, head별)
BATCHED = [("attn.qk^T", NH, S, HD, S), ("attn.av", NH, S, S, HD)]


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="eval_results/gemm_fp16_bf16.json")
    args = ap.parse_args()

    print(f"[GPU] {torch.cuda.get_device_name(0)}  "
          f"capability={'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"[torch] {torch.__version__}  cuBLAS TF32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"[형상] S={S} H={H} I={I} heads={NH}\n")

    dts = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    rows, res = [], {}

    for name, M, K, N in SHAPES:
        flops = 2.0 * M * K * N
        entry = {}
        for dn, dt in dts.items():
            a = torch.randn(M, K, device="cuda", dtype=dt)
            b = torch.randn(K, N, device="cuda", dtype=dt)
            ms = bench(lambda: torch.matmul(a, b), args.iters)
            entry[dn] = {"ms": ms, "tflops": flops / (ms * 1e-3) / 1e12}
            del a, b
            torch.cuda.empty_cache()
        entry["bf16_over_fp16"] = entry["bfloat16"]["ms"] / entry["float16"]["ms"]
        res[name] = entry
        rows.append((name, f"{M}x{K}x{N}", entry))

    for name, B, M, K, N in BATCHED:
        flops = 2.0 * B * M * K * N
        entry = {}
        for dn, dt in dts.items():
            if dn == "float32" and B * M * max(K, N) > 3e8:
                continue  # fp32는 메모리가 커서 건너뛴다
            a = torch.randn(B, M, K, device="cuda", dtype=dt)
            b = torch.randn(B, K, N, device="cuda", dtype=dt)
            ms = bench(lambda: torch.bmm(a, b), max(args.iters // 5, 10))
            entry[dn] = {"ms": ms, "tflops": flops / (ms * 1e-3) / 1e12}
            del a, b
            torch.cuda.empty_cache()
        entry["bf16_over_fp16"] = entry["bfloat16"]["ms"] / entry["float16"]["ms"]
        res[name] = entry
        rows.append((name, f"{B}x{M}x{K}x{N}", entry))

    print(f"{'GEMM':<14} {'형상':<22} {'fp16':>18} {'bf16':>18} {'bf16/fp16':>10}")
    print("-" * 86)
    for name, shape, e in rows:
        f, b = e["float16"], e["bfloat16"]
        print(f"{name:<14} {shape:<22} "
              f"{f['ms']:>8.3f} ms {f['tflops']:>6.1f}T "
              f"{b['ms']:>8.3f} ms {b['tflops']:>6.1f}T "
              f"{e['bf16_over_fp16']:>9.3f}배")

    ratios = [e["bf16_over_fp16"] for _, _, e in rows]
    geo = float(torch.tensor(ratios).log().mean().exp())
    print("-" * 86)
    print(f"기하평균 bf16/fp16 = {geo:.3f}배   (TensorRT 엔진 실측: 1.413배)")
    print()
    print("=" * 78)
    if abs(geo - 1.0) <= 0.05:
        print("판정: **순수 GEMM에서 두 형식은 동등하다.** 따라서 TensorRT의 1.41배는")
        print("  형식의 산술 비용이 아니라 **런타임 커널 커버리지**에서 온다.")
        print("  사전 등록한 대로 §7.2의 '형식 선택 자체가 교환'을 철회한다.")
    elif geo > 1.05:
        print(f"판정: bf16이 순수 GEMM에서도 {geo:.2f}배 느리다. 현재 서술을 유지한다.")
    else:
        print(f"판정: bf16이 오히려 빠르다({geo:.3f}배). 측정 조건을 먼저 의심할 것.")
    print("=" * 78)

    out = {"gpu": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability(0)),
           "torch": torch.__version__, "iters": args.iters,
           "shapes": {"S": S, "H": H, "I": I, "heads": NH},
           "trt_engine_ratio": 1.4126, "geomean_bf16_over_fp16": geo,
           "results": res}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
