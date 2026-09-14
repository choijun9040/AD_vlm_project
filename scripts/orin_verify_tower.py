"""
Orin Nano에서 비전 타워 TensorRT 엔진의 fp16 붕괴율을 잰다 (논문 7장).
==========================================================================
A100 기준값(원본 해상도 1,440,000, 같은 250장):
    student_baseline_v2  94.8% 붕괴 / student_full  0.0%

**전처리를 A100과 동일하게 유지하는 것이 이 스크립트의 핵심이다.** 붕괴는 활성 크기에
의존하고 활성 크기는 해상도에 의존하므로, 전처리가 조금만 달라도 비교가 무의미해진다.
그래서 보드에서도 같은 `AutoProcessor`를 쓴다(CPU만 쓰므로 GPU 메모리와 무관하다).

의존성 (보드):
    pip install transformers pillow numpy    # 전처리 전용, CPU
    TensorRT는 JetPack에 포함돼 있다. cuda-python 또는 pycuda 중 하나 필요.

실행:
    python3 orin_verify_tower.py --engine b_fp16.plan --images images/ \
        --list img250.txt --max_pixels 1440000 --tag baseline_v2_fp16
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"


# --------------------------------------------------------------------------
# TensorRT 실행기 — cuda-python과 pycuda 어느 쪽이 있어도 동작하게 한다
# --------------------------------------------------------------------------
class Runner:
    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"엔진을 읽지 못했다: {engine_path}")
        self.ctx = self.engine.create_execution_context()

        self.i_name = self.engine.get_tensor_name(0)
        self.o_name = self.engine.get_tensor_name(1)
        self.i_shape = tuple(self.engine.get_tensor_shape(self.i_name))
        self.o_shape = tuple(self.engine.get_tensor_shape(self.o_name))
        self.i_dtype = trt.nptype(self.engine.get_tensor_dtype(self.i_name))
        self.o_dtype = trt.nptype(self.engine.get_tensor_dtype(self.o_name))
        print(f"  입력 {self.i_name}{self.i_shape} {np.dtype(self.i_dtype).name}"
              f"  출력 {self.o_name}{self.o_shape} {np.dtype(self.o_dtype).name}")

        self._init_cuda()

    def _init_cuda(self):
        try:
            from cuda import cudart
            self._api, self._rt = "cuda-python", cudart
            nb_i = int(np.prod(self.i_shape)) * np.dtype(self.i_dtype).itemsize
            nb_o = int(np.prod(self.o_shape)) * np.dtype(self.o_dtype).itemsize
            self.d_in = cudart.cudaMalloc(nb_i)[1]
            self.d_out = cudart.cudaMalloc(nb_o)[1]
            self.stream = cudart.cudaStreamCreate()[1]
        except ImportError:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            self._api, self._rt = "pycuda", cuda
            self.d_in = cuda.mem_alloc(int(np.prod(self.i_shape))
                                       * np.dtype(self.i_dtype).itemsize)
            self.d_out = cuda.mem_alloc(int(np.prod(self.o_shape))
                                        * np.dtype(self.o_dtype).itemsize)
            self.stream = cuda.Stream()
        print(f"  CUDA 바인딩: {self._api}")

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=self.i_dtype)
        out = np.empty(self.o_shape, dtype=self.o_dtype)
        if self._api == "cuda-python":
            rt = self._rt
            rt.cudaMemcpyAsync(self.d_in, x.ctypes.data, x.nbytes,
                               rt.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            self.ctx.set_tensor_address(self.i_name, int(self.d_in))
            self.ctx.set_tensor_address(self.o_name, int(self.d_out))
            self.ctx.execute_async_v3(self.stream)
            rt.cudaMemcpyAsync(out.ctypes.data, self.d_out, out.nbytes,
                               rt.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
            rt.cudaStreamSynchronize(self.stream)
        else:
            cuda = self._rt
            cuda.memcpy_htod_async(self.d_in, x, self.stream)
            self.ctx.set_tensor_address(self.i_name, int(self.d_in))
            self.ctx.set_tensor_address(self.o_name, int(self.d_out))
            self.ctx.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
            self.stream.synchronize()
        return out


def peak_mem_mib():
    """보드는 CPU/GPU 메모리를 공유하므로 시스템 전체 사용량을 본다."""
    try:
        with open("/proc/meminfo") as f:
            m = {l.split(":")[0]: int(l.split()[1]) for l in f}
        return (m["MemTotal"] - m["MemAvailable"]) / 1024
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="trtexec가 만든 .plan")
    ap.add_argument("--images", required=True, help="이미지가 든 디렉터리")
    ap.add_argument("--list", default=None,
                    help="쓸 파일명 목록(한 줄에 하나). 없으면 디렉터리 전체")
    ap.add_argument("--max_pixels", type=int, required=True,
                    help="**엔진을 만든 해상도와 반드시 같아야 한다**")
    ap.add_argument("--min_pixels", type=int, default=3136)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.images)
    if args.list:
        names = [l.strip() for l in open(args.list) if l.strip()]
        paths = [root / n for n in names]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"이미지 {len(missing)}장이 없다. 예: {missing[0]}")
    else:
        paths = sorted(root.glob("*.jpg"))
    print(f"[대상] {len(paths)}장, max_pixels={args.max_pixels}")

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE, max_pixels=args.max_pixels,
                                         min_pixels=args.min_pixels)

    print(f"[엔진] {args.engine}")
    run = Runner(args.engine)

    for _ in range(args.warmup):
        run(np.zeros(run.i_shape, dtype=run.i_dtype))

    base_mem = peak_mem_mib()
    nan_imgs, lat, peak, per_image = [], [], base_mem, []
    for i, p in enumerate(paths, 1):
        enc = proc.image_processor(images=[Image.open(p).convert("RGB")],
                                   return_tensors="np")
        pv = enc["pixel_values"]
        if tuple(pv.shape) != run.i_shape:
            raise SystemExit(
                f"형상 불일치: 이미지 {pv.shape} vs 엔진 {run.i_shape}. "
                f"--max_pixels가 엔진 빌드 때와 다르다.")
        t0 = time.perf_counter()
        out = run(pv)
        lat.append((time.perf_counter() - t0) * 1000)

        bad = bool(~np.isfinite(out).all())
        if bad:
            nan_imgs.append(p.name)
        finite = out[np.isfinite(out)]
        per_image.append({"image": p.name, "nan": bad,
                          "max_abs": float(np.abs(finite).max()) if finite.size else None})
        peak = max(peak, peak_mem_mib())
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}  붕괴 {len(nan_imgs)}  "
                  f"지연 중앙값 {np.median(lat):.0f} ms")

    lat = np.array(lat)
    finite_max = [r["max_abs"] for r in per_image if r["max_abs"] is not None]
    res = {
        "tag": args.tag, "engine": args.engine, "n": len(paths),
        "max_pixels": args.max_pixels,
        "nan_rate": len(nan_imgs) / len(paths),
        "nan_count": len(nan_imgs),
        "latency_ms": {"p50": float(np.median(lat)),
                       "p95": float(np.percentile(lat, 95)),
                       "min": float(lat.min())},
        "mem_mib": {"baseline": base_mem, "peak": peak, "delta": peak - base_mem},
        "max_abs_p50": float(np.median(finite_max)) if finite_max else None,
        "headroom_p50": (65504.0 / float(np.median(finite_max))) if finite_max else None,
        "nan_images": nan_imgs[:20],
    }
    print("\n" + "=" * 60)
    print(f"붕괴율      {res['nan_rate']*100:.1f}%  ({res['nan_count']}/{res['n']})")
    print(f"지연        p50 {res['latency_ms']['p50']:.0f} ms / "
          f"p95 {res['latency_ms']['p95']:.0f} ms")
    print(f"메모리 증가  {res['mem_mib']['delta']:.0f} MiB (피크 {res['mem_mib']['peak']:.0f})")
    if res["headroom_p50"]:
        print(f"여유(유한값) p50 {res['headroom_p50']:.2f}배  "
              f"(max|act| {res['max_abs_p50']:.0f})")
    print("=" * 60)
    print("A100 기준값(원본 해상도): baseline_v2 94.8% / full 0.0%")

    out = Path(args.out or f"orin_{args.tag}.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
