"""
보드의 전처리가 A100과 비트 단위로 같은지 확인한다 (측정 전에 반드시 실행).
기준 해시는 A100 서버에서 transformers 4.49.0으로 만들었다(2026-09-29).
다르면 측정하지 말 것 — 붕괴는 활성 크기, 활성 크기는 입력에 의존한다.

    HF_HOME=$PWD/hf_cache_processor HF_HUB_OFFLINE=1 python3 orin_check_preproc.py
"""
import hashlib, json, sys
import numpy as np
from PIL import Image
import transformers
from transformers import AutoProcessor

ref = json.load(open("preproc_ref.json"))
print(f"transformers {transformers.__version__} (기준 {ref['transformers']})")
ok = True
for mp, items in ref["hashes"].items():
    p = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",
                                      max_pixels=int(mp), min_pixels=3136)
    for name, want in items.items():
        pv = p.image_processor(images=[Image.open(f"images/{name}").convert("RGB")],
                               return_tensors="np")["pixel_values"]
        got = hashlib.sha256(np.ascontiguousarray(pv).tobytes()).hexdigest()
        same = got == want["sha256"] and list(pv.shape) == want["shape"]
        ok &= same
        print(f"  {mp:>8} {name[:40]:40} {tuple(pv.shape)}  {'일치' if same else '**불일치**'}")
print("통과 — 측정해도 된다" if ok else "**불일치 — transformers==4.49.0인지, 캐시가 꾸러미 것인지 확인**")
sys.exit(0 if ok else 1)
