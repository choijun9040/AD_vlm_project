"""
A* — Hugging Face DebugUnderflowOverflow를 문서 그대로: 모델 **전체**에 걸고 실행, 처음 감지에서 멈춘다. 2026-11-04.
사전 등록: thesis_outline_20260910.md §6 개정 16 (A*, 결과 보기 전).

문서 사용법: `debug_overflow = DebugUnderflowOverflow(model)` 뒤 평소대로 실행 — inf/nan을 처음 만나면 직전 프레임을 출력하고 멈춘다.
DriveLM 검증 CAM_FRONT 799장을 표준 순서로, 이미지마다 첫 NuScenes-QA 문항과 함께 순전파(fp16, 원본 해상도)한다. 기록: 처음 감지한 이미지 번호와
그 프레임에서 처음 inf/nan이 나온 모듈(프레임의 모듈 머리 줄 아래 「최소 최대 변수」 줄), 감지가 없으면 「799장 동안 경고 없음」.

    python scripts/cmp_debug_overflow_exact.py --checkpoint checkpoints/student_baseline_v2/epoch_1 --tag baseline_unfixed --out eval_results/cmp/ax_baseline_unfixed.json
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import awq_compat                                   # noqa: E402
from dataloader import build_token_to_images       # noqa: E402

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
VAL = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"
QA = "data/nuscenes_qa/NuScenes_val_questions.json"


def first_inf_module(dump):
    """프레임: 「<접두 공백> 모듈이름 클래스」 머리 줄 + 「최소 최대 변수」 줄. 처음 inf/nan이 나온 변수 줄의 모듈과 변수를 돌려준다."""
    cur = None
    for line in dump.splitlines():
        t = line.split()
        if len(t) >= 3 and (t[1].lower() in ("inf", "nan") or t[0].lower() in ("inf", "nan")):
            return cur, t[2]
        if len(t) == 2 and not t[0][0].isdigit():
            cur = t[0]
        elif len(t) >= 2 and line.startswith(" " * 10):
            cur = t[0]
    return cur, None


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vision_state", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=799)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    awq_compat.patch()
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.debug_utils import DebugUnderflowOverflow

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16, attn_implementation="eager")
    if a.checkpoint.lower() != "none":
        model = PeftModel.from_pretrained(model, a.checkpoint).merge_and_unload()
    if a.vision_state:
        import headroom_ext as HX
        sd = torch.load(a.vision_state, map_location="cpu")
        fp, ep = Path(a.vision_state + ".fix.json"), Path(a.vision_state + ".eps.json")
        side = json.loads(fp.read_text()) if fp.exists() else ({"eps": json.loads(ep.read_text())} if ep.exists() else {})
        HX.load_fixed(model.visual, sd, side)
        model.visual.half()
    model = model.to("cuda").eval()
    proc = AutoProcessor.from_pretrained(BASE, max_pixels=1440000, min_pixels=3136)
    tok = build_token_to_images(VAL)
    first_q = {}
    for q in json.load(open(QA))["questions"]:
        first_q.setdefault(q["sample_token"], q["question"])
    items = [(t, str(v["CAM_FRONT"]), first_q[t]) for t, v in tok.items() if "CAM_FRONT" in v and t in first_q][:a.limit]

    dbg = DebugUnderflowOverflow(model)                 # 문서 기본값
    res = {"tag": a.tag, "checkpoint": a.checkpoint, "vision_state": a.vision_state, "n_items": len(items),
           "detected": False, "image_index": None, "module": None, "variable": None}
    for i, (_t, p, q) in enumerate(items):
        msgs = [{"role": "user", "content": [{"type": "image", "image": p}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[Image.open(p).convert("RGB")], return_tensors="pt").to("cuda")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), torch.no_grad():
                model(**inp)
        except ValueError as e:
            if "inf/nan detected" not in str(e):
                raise
            mod, var = first_inf_module(buf.getvalue())
            res.update({"detected": True, "image_index": i, "module": mod, "variable": var, "dump_tail": buf.getvalue()[-4000:]})
            print(f"[{a.tag}] 처음 감지: 이미지 {i} · 모듈 {mod} · 변수 {var} — 문서대로 여기서 멈춘다", flush=True)
            break
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(items)} 감지 없음", flush=True)
    if not res["detected"]:
        print(f"[{a.tag}] {len(items)}장 동안 경고 없음", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
