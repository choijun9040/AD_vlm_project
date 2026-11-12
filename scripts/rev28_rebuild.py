"""개정 28 — 같은 설정으로 TensorRT 엔진을 다시 빌드하면 공식 Qwen 무교정 fp16 상실이 달라지는가. 2026-11-04, 결과 보기 전 등록(23:37 KST).
① 무교정 그래프 5회 재빌드 ② 기존 두 엔진 각 2회 실행 ③ (R) 그래프 2회 재빌드 ④ 엔진 검사기로 빌드 간 층별 커널 대조. 입력은 RealWorldQA 952×1456 465장(lat/inputs465.npz).
venv_ort. A100에 다른 GPU 작업이 없을 때 돈다(감시는 rev_idle_chain.sh가 한다).
"""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cmp_trt_eval import runner                     # noqa: E402

import os
R = Path(os.environ.get("RDIR", "/dev/shm/rwqa952")); D = R / "rebuild"; D.mkdir(exist_ok=True)   # 개정 30: DriveLM은 RDIR=/dev/shm/dlm799 GDIR=onnx GBASE=tower_official_dlm1596h
G = R / os.environ.get("GDIR", "onnx2")
GBASE = os.environ.get("GBASE", "tower_official_rwqa952h")
NPZ = Path(os.environ.get("NPZ", str(R / "lat" / "inputs465.npz")))
A_DIR = R / os.environ.get("A_PLANS", "plans2")
OUT = Path(os.environ.get("OUT", "eval_results/cmp/rev28_rebuild.json"))


def build(onnx, plan):
    import tensorrt as trt
    lg = trt.Logger(trt.Logger.WARNING); rt = trt.Runtime(lg)
    b = trt.Builder(lg)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(net, lg)
    assert p.parse_from_file(str(onnx)), "ONNX 파싱 실패"
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 24 << 30)        # cmp_trt_eval.py·trt_precision_probe.py와 같다
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED                  # 검사기 기록용 — 커널 선택과 무관
    t0 = time.time(); blob = bytes(b.build_serialized_network(net, cfg)); dt = time.time() - t0
    Path(plan).write_bytes(blob)
    eng = rt.deserialize_cuda_engine(blob)
    insp = eng.create_engine_inspector()
    info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
    return eng, hashlib.sha256(blob).hexdigest(), info, dt


def load(plan):
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    return rt.deserialize_cuda_engine(Path(plan).read_bytes())


def lost_set(e1, e2, z, keys):
    r1, r2 = runner(e1), runner(e2)
    bad = []
    with torch.no_grad():
        for k, key in enumerate(keys):
            y = r2(r1(torch.from_numpy(z[key])))
            torch.cuda.synchronize()
            if not torch.isfinite(y).all():
                bad.append(k)
    return bad


def tactics(info):
    out = {}
    for l in info.get("Layers", []):
        if isinstance(l, dict):
            out[l.get("Name", "?")] = l.get("TacticName", l.get("Tactic", "")) + "|" + str(l.get("LayerType", ""))
    return out


def main():
    z = np.load(NPZ)
    keys = sorted(k for k in z.files if not k.startswith("_"))
    res = {"n": len(keys), "rebuild_unfixed": [], "rebuild_resid": [], "rerun_existing": {}}
    # ② 기존 두 엔진, 각 2회
    for tag, d in (("A_cmp_trt_eval", A_DIR), ("B_trt_precision_probe", R / "lat" / "plans")):
        p1 = sorted(d.glob(f"{GBASE}_p1*.plan")); p2 = sorted(d.glob(f"{GBASE}_p2*.plan"))
        if not p1 or not p2:
            res["rerun_existing"][tag] = "엔진 파일 없음"; continue
        e1, e2 = load(p1[0]), load(p2[0])
        runs = [lost_set(e1, e2, z, keys) for _ in range(2)]
        res["rerun_existing"][tag] = {"plans": [str(p1[0]), str(p2[0])], "lost": runs, "same": runs[0] == runs[1]}
        print(f"[기존 {tag}] 실행 2회 상실 {[len(r) for r in runs]} 같음 {runs[0] == runs[1]}", flush=True)
        del e1, e2; torch.cuda.empty_cache()
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    # ①·③ 재빌드
    for kind, base, n in (("unfixed", GBASE, 5), ("resid", GBASE + "_resid", 2)):
        for i in range(n):
            e1, h1, info1, t1 = build(G / f"{base}_p1.onnx", D / f"{kind}{i}_p1.plan")
            e2, h2, info2, t2 = build(G / f"{base}_p2.onnx", D / f"{kind}{i}_p2.plan")
            lost = lost_set(e1, e2, z, keys)
            (D / f"{kind}{i}_p2.inspect.json").write_text(json.dumps(info2))
            res[f"rebuild_{kind}"].append({"i": i, "sha256": [h1, h2], "build_s": [t1, t2], "lost": lost, "tactics_p2": tactics(info2)})
            print(f"[재빌드 {kind} {i}] 상실 {len(lost)} · 계획 해시 {h1[:10]} {h2[:10]} · 빌드 {t1:.0f}+{t2:.0f}초", flush=True)
            del e1, e2; torch.cuda.empty_cache()
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    # ④ 빌드 간 커널 차이(무교정)
    tl = [b["tactics_p2"] for b in res["rebuild_unfixed"]]
    if tl:
        layers = sorted(set().union(*[t.keys() for t in tl]))
        diff = [l for l in layers if len({t.get(l) for t in tl}) > 1]
        res["unfixed_p2_layers_with_different_tactics"] = {"n_layers": len(layers), "n_diff": len(diff), "examples": diff[:30]}
        sets = [tuple(b["lost"]) for b in res["rebuild_unfixed"]]
        res["unfixed_lost_summary"] = {"counts": [len(s) for s in sets], "union": len(set().union(*map(set, sets))),
                                       "intersection": len(set.intersection(*map(set, sets)))}
        print(f"[커널] p2 층 {len(layers)}개 중 빌드마다 다른 층 {len(diff)}개 · 상실 수 {[len(s) for s in sets]}", flush=True)
    for b in res["rebuild_unfixed"] + res["rebuild_resid"]:
        b.pop("tactics_p2", None)                       # 큰 표는 inspect.json에
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
