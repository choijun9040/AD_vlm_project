"""⑥에서 성공으로 기록된 모델들이 **실제로 가중치를 로드했는지** 확인한다.

2026-09-15에 드러난 함정: 체크포인트 키 규약이 이 transformers 버전과 다르면
`from_pretrained`가 경고만 찍고 **랜덤 초기화 모델**을 돌려준다. 로드는 "성공"하고
비전 타워도 찾아지며 숫자도 나온다 — 그 숫자가 난수일 뿐이다.
따라서 이미 기록된 여유값도 재검증해야 한다.

판정은 두 가지를 함께 본다.
  (1) norm 계열 1차원 파라미터가 정확히 1.0인가 (랜덤 초기화의 특징)
  (2) `from_pretrained`가 "newly initialized"를 경고했는가 (로그 캡처)

실행:
    python scripts/verify_family_weights.py --cleanup
"""

import argparse
import contextlib
import io
import json
import shutil
import sys
import warnings
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
import awq_compat
from profile_other_vlm_families import TARGETS, find_vision_tower

RESULT = "eval_results/other_vlm_families_profile.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--out", default="eval_results/family_weight_verification.json")
    args = ap.parse_args()

    awq_compat.patch()
    import transformers as HF
    from transformers import AutoConfig

    done = json.load(open(RESULT))
    targets = [(t, r, n) for t, r, n in TARGETS if "headroom_p50" in done.get(t, {})]
    print(f"검증 대상: 여유값이 기록된 {len(targets)}개\n")

    out = {}
    for tag, repo, note in targets:
        print(f"=== {tag} — {repo}", flush=True)
        rec = {"repo": repo, "recorded_headroom": done[tag]["headroom_p50"]}
        buf = io.StringIO()
        try:
            trc = "[remote_code]" in note
            cfg = AutoConfig.from_pretrained(repo, trust_remote_code=trc)
            if isinstance(getattr(cfg, "text_config", None), dict):
                delattr(cfg, "text_config")
            order = [a for a in (getattr(cfg, "architectures", None) or [])
                     if getattr(HF, a, None) is not None]
            order += ["AutoModelForVision2Seq", "AutoModelForImageTextToText", "AutoModel"]
            model = None
            with contextlib.redirect_stderr(buf), warnings.catch_warnings():
                warnings.simplefilter("always")
                for cn in order:
                    cls = getattr(HF, cn, None)
                    if cls is None:
                        continue
                    try:
                        kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=trc)
                        if cn != "AutoModel":
                            kw["config"] = cfg
                        model = cls.from_pretrained(repo, **kw)
                        find_vision_tower(model)
                        rec["loader"] = cn
                        break
                    except Exception:
                        model = None
            if model is None:
                rec["verdict"] = "로드 실패"
                print("  로드 실패"); out[tag] = rec; continue

            tower = find_vision_tower(model)
            norms = [(n, p) for n, p in tower.named_parameters()
                     if p.dim() == 1 and ("norm" in n.lower() or "ln" in n.lower())]
            exact_one = sum(1 for _, p in norms[:24]
                            if bool((p.detach().float() == 1.0).all()))
            allp = [p for _, p in tower.named_parameters() if p.dim() > 1]
            std = float(torch.cat([p.detach().float().flatten()[:2000] for p in allp[:8]]).std())
            warned = "newly initialized" in buf.getvalue()

            rec.update({"n_norm_params": len(norms), "norm_exactly_one": exact_one,
                        "weight_std_sample": round(std, 5),
                        "warned_newly_initialized": warned})
            bad = warned or (norms and exact_one >= max(1, min(24, len(norms)) // 2))
            rec["verdict"] = "**가중치 미로드 의심**" if bad else "정상 로드"
            print(f"  로더={rec.get('loader')}  norm==1.0 {exact_one}/{min(24,len(norms))}  "
                  f"std={std:.4f}  경고={warned}  → {rec['verdict']}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            rec["verdict"] = f"검증 실패: {type(e).__name__}: {str(e)[:120]}"
            print(f"  {rec['verdict']}")
        out[tag] = rec
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        if args.cleanup:
            import os
            cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            d = Path(cache) / "hub" / ("models--" + repo.replace("/", "--"))
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print("  캐시 삭제")

    print(f"\n{'태그':<22}{'기록 여유':>10}  판정")
    for t, r in out.items():
        print(f"{t:<22}{r['recorded_headroom']:>9.2f}배  {r['verdict']}")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
