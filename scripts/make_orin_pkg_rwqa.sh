#!/bin/bash
# Orin 꾸러미 — 공식 Qwen2.5-VL-3B, RealWorldQA 952×1456 형상 465문항 (사전 등록 개정 23, 2026-11-04). A100 서버에서 돈다.
# 디스크가 부족해 모든 산출물을 램 디스크(/dev/shm)에 둔다. 먼저 끝나 있어야 하는 것:
#   /dev/shm/rwqa952/{images,img465.txt,paths465.txt}   (prep_rwqa_orin_images.py)
#   /dev/shm/rwqa952/onnx/tower_official_rwqa952h{,_resid}_p{1,2}.onnx(.data)   (export_vision_tower_onnx.py)
#   eval_results/cmp/rwqa_trt_a100.json   (cmp_trt_eval.py — A100 fp16 엔진 기준, 없으면 기준 없이 묶는다)
# 만드는 것: /dev/shm/rwqa952/orin_pkg_rwqa_<날짜>.tar.gz (스크립트·이미지·캐시·체크섬), orin_onnx_rwqa_<날짜>.tar (ONNX 4조각), SHA256SUMS_rwqa
set -u
cd ${REPO:-/workspace/vlm_project_2}
PY=${PY:-/workspace/venv_main/bin/python}
R=/dev/shm/rwqa952; G=$R/onnx2; S=$R/orin_pkg; STAMP=$(date +%Y%m%d)
log(){ echo "[$(date '+%F %T')] $*"; }
ONNX=""
for m in tower_official_rwqa952h tower_official_rwqa952h_resid; do for p in p1 p2; do ONNX="$ONNX ${m}_$p.onnx ${m}_$p.onnx.data"; done; done
miss=0; for f in $ONNX; do [ -f $G/$f ] || { echo "  없음: $G/$f"; miss=1; }; done
[ $miss = 0 ] || { log "**ONNX가 없다 — 내보내기를 먼저 끝낼 것**"; exit 1; }
grep -q "official_residual_rwqa.pt" logs/rwqa_export_resid.log && grep -q "옆 파일 eps" logs/rwqa_export_resid.log \
  || { log "**(R) 그래프가 official_residual_rwqa.pt + eps 옆 파일에서 나왔다는 기록이 없다 — 중단**"; exit 1; }

rm -rf $S; mkdir -p $S/a100_ref
cp -r $R/images $S/; cp $R/img465.txt $S/
$PY - $S <<'PY'
import json, sys
from pathlib import Path
S = Path(sys.argv[1]); names = (S / "img465.txt").read_text().split()
(S / "img465_scenes.json").write_text(json.dumps({n: "rwqa" for n in names}, indent=0))   # 장면 대신 데이터셋 이름(장면 군집 없음)
PY
cat > $S/scene_map.py <<'PY'
"""보드용 장면 매핑 대용 — RealWorldQA에는 장면 정보가 없다. 모든 이미지를 한 군집 'rwqa'로."""
import json
from pathlib import Path


def image_to_scene():
    return json.loads((Path(__file__).resolve().parent / "img465_scenes.json").read_text())
PY
cp scripts/orin/orin_verify_tower.py scripts/orin/orin_build_engine.py scripts/orin/orin_check_preproc.py scripts/orin/orin_run_rwqa.sh $S/
cp scripts/ORIN_RWQA_README.md $S/README.md
cp -r onnx/hf_cache_processor $S/
# 전처리 기준 해시 — 이 꾸러미의 캐시·이미지로 A100에서 만든다(앞·중간·끝 5장)
( cd $S && HF_HOME=$PWD/hf_cache_processor HF_HUB_OFFLINE=1 $PY - <<'PY'
import hashlib, json, numpy as np, transformers
from PIL import Image
from transformers import AutoProcessor
names = open("img465.txt").read().split(); pick = [names[i] for i in (0, 1, len(names) // 2, len(names) - 2, len(names) - 1)]
p = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=12845056, min_pixels=3136)
h = {}
for n in pick:
    pv = p.image_processor(images=[Image.open(f"images/{n}").convert("RGB")], return_tensors="np")["pixel_values"]
    h[n] = {"sha256": hashlib.sha256(np.ascontiguousarray(pv).tobytes()).hexdigest(), "shape": list(pv.shape)}
json.dump({"transformers": transformers.__version__, "hashes": {"12845056": h}}, open("preproc_ref.json", "w"), indent=1)
print("  전처리 기준 해시", len(h), "장", next(iter(h.values()))["shape"])
PY
) || { log "**전처리 기준 해시 만들기 실패**"; exit 1; }
( cd $S && HF_HOME=$PWD/hf_cache_processor HF_HUB_OFFLINE=1 $PY orin_check_preproc.py ) || { log "**꾸러미 전처리 점검 실패 — 중단**"; exit 1; }
# A100 기준(보드 스크립트가 끝에 나란히 출력) — cmp_trt_eval.py 결과를 보드 형식으로
$PY - $S <<'PY'
import json, sys, os
S = sys.argv[1]; f = "eval_results/cmp/rwqa_trt_a100.json"
if not os.path.exists(f):
    print("  (A100 기준 없음)"); sys.exit(0)
d = json.load(open(f))
for tag, m in (("official", "official"), ("resid", "resid")):
    c = d["configs"].get(tag)
    if c:
        json.dump({"n_images": d["n"], "modes": {"typed": {"nan_rate": c["loss"] / d["n"]}}, "source": f},
                  open(f"{S}/a100_ref/trt_rwqa952h_typed_{m}.json", "w"), indent=1)
        print(f"  A100 기준 {tag}: 상실 {c['loss']}/{d['n']}")
PY
chmod +x $S/orin_run_rwqa.sh
( cd $S && find . -type f ! -name SHA256SUMS_pkg ! -name SHA256SUMS_onnx | sed 's|^\./||' | sort | xargs -d '\n' sha256sum > SHA256SUMS_pkg )
( cd $G && sha256sum $ONNX > $S/SHA256SUMS_onnx )
PKG=$R/orin_pkg_rwqa_$STAMP.tar.gz; OT=$R/orin_onnx_rwqa_$STAMP.tar
tar czf $PKG -C $R orin_pkg
tar cf $OT -C $G --transform 's,^,orin_pkg/,' $ONNX
( cd $R && sha256sum $(basename $PKG) $(basename $OT) > SHA256SUMS_rwqa && cat SHA256SUMS_rwqa )
ls -la $PKG $OT | awk '{printf "  %s  %.2f GB\n", $NF, $5/2^30}'
log "꾸러미 완료 — 보드 절차는 scripts/ORIN_RWQA_README.md"
