#!/bin/bash
# Orin 꾸러미 만들기 — 평가 데이터 전부(799장)로 통일 (사전 등록 4fddeb4, 실행 순서 6). A100 서버에서 돈다.
#
# run_full799_s4_trt.sh가 onnx/full799/에 남긴 fp16 그래프(802816h·401408h × 기준선·Full·799장 교정본, 분할 p1/p2)와
# DriveLM 검증 CAM_FRONT 799장(표준 순서), 실행 스크립트, 전처리 캐시·기준 해시, 체크섬을 묶는다.
# 6장 [orin-result]에 실린 조건이 그대로다 — 원본 해상도(nativeh)는 보드에서 강타입으로도 빌드되지 않았으므로 넣지 않는다.
#
# 만드는 것 (onnx/ 아래):
#   orin_pkg_full799_<날짜>.tar.gz   스크립트·이미지 799장·캐시·체크섬 (~120 MB)
#   orin_onnx_full799_<날짜>.tar     ONNX 24개 (~8.6 GB, 압축 안 함 — fp16 가중치는 gzip으로 거의 줄지 않는다)
#   SHA256SUMS_full799              위 두 묶음의 체크섬
# 두 묶음을 보드의 같은 디렉터리에서 풀면 둘 다 orin_pkg/ 하나로 들어간다. ONNX_TAR=0이면 ONNX 묶음을 만들지 않는다
# (onnx/full799/의 ONNX 24개를 rsync로 orin_pkg/에 직접 옮길 때 — 디스크 8.6 GB 절약).
#
#   bash scripts/make_orin_pkg_full799.sh
set -u
cd ${REPO:-/workspace/vlm_project_2}
PY=${PY:-/workspace/venv_main/bin/python}
G=onnx/full799; S=$G/orin_pkg; STAMP=$(date +%Y%m%d)
log(){ echo "[$(date '+%F %T')] $*"; }

# ── 1. 그래프 — 6장 보드 측정과 같은 판본 (fp16 그래프, 분할 16, 어텐션 나눠 계산), 교정본은 799장 가중치 ──────
ONNX=""
for RES in 802816h 401408h; do
  for m in tower_baseline_v2_$RES tower_full_$RES tower_baseline_v2_${RES}_fixa; do
    for p in p1 p2; do ONNX="$ONNX ${m}_$p.onnx ${m}_$p.onnx.data"; done
  done
done
miss=0; for f in $ONNX; do [ -f $G/$f ] || { echo "  없음: $G/$f"; miss=1; }; done
[ $miss = 0 ] || { log "**그래프가 없다 — run_full799_s4_trt.sh의 802816h·401408h 내보내기를 먼저 끝낼 것**"; exit 1; }
# 교정본이 799장 가중치에서 나왔는지 — 내보내기 로그로 확인한다
for RES in 802816h 401408h; do
  grep -q "baseline_v2_fixa_native_tower_full799.pt" logs/full799_trt/export_tower_baseline_v2_${RES}_fixa.log 2>/dev/null \
    || { log "**tower_baseline_v2_${RES}_fixa 가 799장 교정 가중치에서 나왔다는 기록이 없다 — 중단**"; exit 1; }
done
log "[1] 그래프 24개 확인 (교정본은 baseline_v2_fixa_native_tower_full799.pt)"

# ── 2. 꾸러미 디렉터리 ─────────────────────────────────────────────────────────────
rm -rf $S; mkdir -p $S/images $S/a100_ref
$PY - $S <<'PY' || { echo "**이미지 목록 만들기 실패**"; exit 1; }
import json, shutil, sys
from pathlib import Path
sys.path.append("scripts")
from dataloader import build_token_to_images
from scene_map import image_to_scene
S = Path(sys.argv[1])
tok = build_token_to_images("data/QA_dataset_nus/v1_1_val_nus_q_only.json")
paths = [Path(v["CAM_FRONT"]) for v in tok.values() if "CAM_FRONT" in v]   # prep_trt_inputs.py와 같은 표준 순서
names = [p.name for p in paths]
assert len(names) == 799 == len(set(names)), f"CAM_FRONT {len(names)}장 (고유 {len(set(names))})"
old = Path("onnx/img250.txt")
if old.exists():   # 이전 꾸러미의 250장은 이 순서의 앞자르기였다 — 순서가 같아야 한다
    assert old.read_text().split() == names[:250], "img250.txt가 표준 순서의 앞 250장과 다르다"
for p in paths:
    shutil.copy2(p, S / "images" / p.name)
(S / "img799.txt").write_text("\n".join(names) + "\n")
sc = image_to_scene()
(S / "img799_scenes.json").write_text(json.dumps({n: sc[n] for n in names}, indent=0))
print(f"  이미지 {len(names)}장, 장면 {len(set(sc[n] for n in names))}개 — img799.txt, img799_scenes.json")
PY
# 장면 매핑 — orin_verify_tower.py --per_image가 같은 디렉터리의 scene_map을 먼저 찾는다. 보드에는 DriveLM JSON이 없으므로
# A100에서 만든 이미지→장면 표를 읽는 작은 대체본을 넣는다(서버의 scripts/scene_map.py와 같은 값).
cat > $S/scene_map.py <<'PY'
"""보드용 장면 매핑 — A100에서 scripts/scene_map.image_to_scene()로 만든 img799_scenes.json을 읽는다."""
import json
from pathlib import Path


def image_to_scene():
    return json.loads((Path(__file__).resolve().parent / "img799_scenes.json").read_text())
PY
cp scripts/orin/orin_verify_tower.py scripts/orin/orin_build_engine.py scripts/orin/orin_check_preproc.py \
   scripts/orin/orin_run_full799.sh $S/
cp scripts/orin/preproc_ref.json $S/
cmp -s scripts/orin/preproc_ref.json onnx/preproc_ref.json || log "  주의: scripts/orin/preproc_ref.json과 onnx/preproc_ref.json이 다르다 — scripts/orin 판을 넣었다"
cp scripts/ORIN_FULL799_README.md $S/README.md
cp -r onnx/hf_cache_processor $S/
# A100 강타입 기준값(799장) — 보드 스크립트가 끝에 나란히 출력한다. 없으면 빈 채로 둔다
for f in eval_results/full799/trt_{802816h,401408h}_typed_{baseline_v2,full,fixa}.json; do
  [ -f $f ] && cp $f $S/a100_ref/ || echo "  (A100 기준 없음: $f)"
done
chmod +x $S/orin_run_full799.sh

# ── 3. 전처리 점검을 A100에서 먼저 — 꾸러미의 캐시·이미지로 기준 해시가 나와야 한다 ─────────────
( cd $S && HF_HOME=$PWD/hf_cache_processor HF_HUB_OFFLINE=1 $PY orin_check_preproc.py ) \
  || { log "**꾸러미 전처리 점검 실패 — 보드에서도 멈출 것이다. 중단**"; exit 1; }
for n in $($PY -c "import json; print(' '.join(next(iter(json.load(open('$S/preproc_ref.json'))['hashes'].values()))))"); do
  grep -qx "$n" $S/img799.txt || { log "**기준 해시 이미지 $n 이 img799.txt에 없다**"; exit 1; }
done
log "[3] 전처리 점검 통과 (A100, 꾸러미 캐시)"

# ── 4. 체크섬 ─────────────────────────────────────────────────────────────────────
( cd $S && find . -type f ! -name SHA256SUMS_pkg ! -name SHA256SUMS_onnx | sed 's|^\./||' | sort | xargs -d '\n' sha256sum > SHA256SUMS_pkg )
( cd $G && sha256sum $ONNX > orin_pkg/SHA256SUMS_onnx )
log "[4] 체크섬: 꾸러미 $(wc -l < $S/SHA256SUMS_pkg)개, ONNX $(wc -l < $S/SHA256SUMS_onnx)개"

# ── 5. 묶기 ───────────────────────────────────────────────────────────────────────
PKG=onnx/orin_pkg_full799_$STAMP.tar.gz
tar czf $PKG -C $G orin_pkg
OUT="$PKG"
if [ "${ONNX_TAR:-1}" = 1 ]; then
  need=$(du -cb $(printf "$G/%s " $ONNX) | tail -1 | cut -f1)
  avail=$(df -B1 --output=avail onnx | tail -1)
  if [ "$avail" -lt $((need + 2*2**30)) ]; then
    log "**디스크 부족 — ONNX 묶음을 만들지 않는다. onnx/full799/의 ONNX 24개를 orin_pkg/로 직접 옮길 것(README 2-b)**"
  else
    OT=onnx/orin_onnx_full799_$STAMP.tar
    tar cf $OT -C $G --transform 's,^,orin_pkg/,' $ONNX
    OUT="$OUT $OT"
  fi
fi
( cd onnx && sha256sum $(for f in $OUT; do basename $f; done) > SHA256SUMS_full799 )
cat onnx/SHA256SUMS_full799
ls -la $OUT | awk '{printf "  %s  %.2f GB\n", $NF, $5/2^30}'
rm -rf $S
log "꾸러미 완료 — 보드 절차는 scripts/ORIN_FULL799_README.md"
