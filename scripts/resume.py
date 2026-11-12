"""중간 저장·이어 하기 (2026-11-05, 사용자 「끊기더라도 손해 없이 재개되도록」). 결과 파일 옆 `<out>.ckpt/`에 단계별 상태를 원자적으로 저장하고,
같은 인자로 다시 실행하면 이어 간다. 인자가 다르면 이전 상태를 버린다. 끝까지 마치면 done()으로 지운다. headroom_ext.PROGRESS(연산 단위 측정의 50장마다 저장)도 여기에 잇는다."""
import json, shutil, sys
from pathlib import Path
import torch

S = {"dir": None}


def setup(out, argv=None):
    d = Path(str(out) + ".ckpt"); key = json.dumps(sys.argv[1:] if argv is None else argv)
    kf = d / "args.json"
    if d.exists() and (not kf.exists() or kf.read_text() != key):
        shutil.rmtree(d); print(f"[이어 하기] 인자가 달라 이전 중간 저장을 지웠다: {d}", flush=True)
    d.mkdir(parents=True, exist_ok=True); kf.write_text(key); S["dir"] = d
    try:
        import headroom_ext as HX
        HX.PROGRESS.update(load=load, save=save)
    except Exception:
        pass


def load(name):
    f = S["dir"] / f"{name}.pt" if S["dir"] else None
    if f is None or not f.exists():
        return None
    print(f"[이어 하기] {name} 불러옴", flush=True)
    return torch.load(f, map_location="cpu", weights_only=False)


def save(name, obj):
    if S["dir"] is None:
        return
    f = S["dir"] / f"{name}.pt"
    torch.save(obj, str(f) + ".tmp"); Path(str(f) + ".tmp").replace(f)


def done():
    if S["dir"] is not None and S["dir"].exists():
        shutil.rmtree(S["dir"])
