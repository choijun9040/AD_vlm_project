#!/usr/bin/env bash
# Orin Nano 실측 일괄 실행 (논문 6장 · 사전 등록 6.4 / 6.4b)
# 다시 실행하면 이미 만든 엔진·결과는 건너뛴다 — 중간에 끊겨도 그대로 재실행하면 된다.
#
#   bash orin_run_all.sh            # 전부
#   bash orin_run_all.sh fp16       # 관행 --fp16 셋만 (가장 중요)
#   bash orin_run_all.sh fp16 802816   # 802,816 해상도 판본 (원본 해상도 OOM 이후, 사전 등록 6.4c)
#   bash orin_run_all.sh fp16 nativec  # 어텐션 나눠 계산 그래프 (6.4f) — 원본 해상도부터
#   bash orin_run_all.sh fp16 nativeh  # fp16 가중치 그래프 (6.4g) — 원본 해상도부터
#   bash orin_run_all.sh typed nativeh # 같은 fp16 그래프를 강타입으로 빌드 (6.4h)
#   TRT_EXTRA="--memPoolSize=workspace:1024" bash orin_run_all.sh fp16 802816   # 빌드 메모리 부족 시
set -u
cd "$(dirname "$0")"
MODE=${1:-all}
RES=${2:-native}
case $RES in
  native) MP=1440000; MODELS="tower_baseline_v2_native tower_full_native tower_baseline_v2_native_fixa" ;;
  802816) MP=802816;  MODELS="tower_baseline_v2_802816 tower_full_802816 tower_baseline_v2_802816_fixa" ;;
  802816s) MP=802816; SPLIT=1   # 블록 16에서 나눈 두 엔진 (6.4d) — 파일은 <모델>_p1/_p2.onnx
           MODELS="tower_baseline_v2_802816s tower_full_802816s tower_baseline_v2_802816s_fixa" ;;
  401408s) MP=401408; SPLIT=1   # 802,816 분할도 OOM — 6.4e
           MODELS="tower_baseline_v2_401408s tower_full_401408s tower_baseline_v2_401408s_fixa" ;;
  # 어텐션을 나눠 계산하는 그래프 (6.4f) — 분할 유지. 원본 해상도부터 시도하고 안 되면 아래로 내린다
  nativec) MP=1440000; SPLIT=1
           MODELS="tower_baseline_v2_nativec tower_full_nativec tower_baseline_v2_nativec_fixa" ;;
  802816c) MP=802816; SPLIT=1
           MODELS="tower_baseline_v2_802816c tower_full_802816c tower_baseline_v2_802816c_fixa" ;;
  401408c) MP=401408; SPLIT=1
           MODELS="tower_baseline_v2_401408c tower_full_401408c tower_baseline_v2_401408c_fixa" ;;
  # fp16 가중치 그래프 (6.4g) — 나눠 계산·분할 유지, 가중치·활성 fp16. 원본 해상도부터
  nativeh) MP=1440000; SPLIT=1
           MODELS="tower_baseline_v2_nativeh tower_full_nativeh tower_baseline_v2_nativeh_fixa" ;;
  802816h) MP=802816; SPLIT=1
           MODELS="tower_baseline_v2_802816h tower_full_802816h tower_baseline_v2_802816h_fixa" ;;
  401408h) MP=401408; SPLIT=1
           MODELS="tower_baseline_v2_401408h tower_full_401408h tower_baseline_v2_401408h_fixa" ;;
  *) echo "해상도는 native, 802816, 802816s, 401408s, nativec, 802816c, 401408c, nativeh, 802816h, 401408h"; exit 1 ;;
esac
BASE=${MODELS%% *}   # fp32 대조군은 기준선만
SPLIT=${SPLIT:-0}
echo "[해상도] $RES (max_pixels $MP)"
TRTEXEC=${TRTEXEC:-$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)}
export HF_HOME=$PWD/hf_cache_processor HF_HUB_OFFLINE=1
mkdir -p results logs

# ── 0. 환경 기록 (6.4b가 요구) ─────────────────────────────────────────────
if [ ! -f results/env.txt ]; then
  { echo "## date";      date -Is
    echo "## jetpack";   cat /etc/nv_tegra_release 2>/dev/null
    echo "## tensorrt";  python3 -c "import tensorrt; print(tensorrt.__version__)" 2>&1
    dpkg -l 2>/dev/null | grep -iE "tensorrt|nvidia-l4t-core" | awk '{print $2, $3}'
    echo "## nvpmodel";  sudo nvpmodel -q 2>&1
    echo "## python";    python3 -c "import transformers, numpy; print('transformers', transformers.__version__, 'numpy', numpy.__version__)" 2>&1
    echo "## mem";       free -h
    echo "## disk";      df -h .
  } > results/env.txt
  echo "[0] 환경 기록 → results/env.txt"
fi

# ── 1. 파일 무결성 · 전처리 일치 — 하나라도 실패하면 멈춘다 ──────────────
# ONNX는 이 디렉터리(orin_pkg/)에, SHA256SUMS는 상위(옮겨 온 곳)에 있다
# 이번 해상도의 모델 파일만 검사한다 — SHA256SUMS에는 여러 해상도가 있고 보드에는 하나씩 옮긴다
NEED=""
for m in $MODELS; do
  if [ "$SPLIT" = 1 ]; then NEED="$NEED ${m}_p1.onnx ${m}_p1.onnx.data ${m}_p2.onnx ${m}_p2.onnx.data"
  else NEED="$NEED $m.onnx"; [ -f $m.onnx.data ] && NEED="$NEED $m.onnx.data"; fi
done
for f in $NEED; do
  [ -f "$f" ] || { echo "**파일 없음: $f — 옮길 것**"; exit 1; }
  grep -q "  $f\$" ../SHA256SUMS || { echo "**SHA256SUMS에 $f 가 없다**"; exit 1; }
done
printf '%s\n' $NEED | sed 's/^/  /;s/$/$/' | grep -f - ../SHA256SUMS | sha256sum -c --quiet - \
  || { echo "**ONNX 체크섬 불일치 — 파일을 다시 옮길 것**"; exit 1; }
echo "[1] ONNX 체크섬 통과"
python3 orin_check_preproc.py || { echo "**전처리 불일치 — 측정 중단**"; exit 1; }

build () {  # $1=모델 $2=조건(fp16|strict|fp32)
  local m=$1 c=$2 plan=$1_$2.plan
  [ -f "$plan" ] && { echo "  [건너뜀] $plan"; return 0; }
  local flags=""
  case $c in
    fp16)   flags="--fp16" ;;
    strict) flags="--fp16 --precisionConstraints=obey --noTF32" ;;
    typed)  flags="--stronglyTyped" ;;   # 강타입 — 정밀도는 그래프 자료형이 정한다 (6.4h)
    fp32)   flags="" ;;
  esac
  echo "  [빌드] $plan ($flags) — 수십 분 걸릴 수 있다"
  # 레이어 정보 JSON — 빌드가 레이어를 ForeignNode로 융합하면 --verbose 로그만으로는
  # 배정 정밀도가 안 보인다(A100 fp16 로그에서 확인). 이 JSON이 입출력 형식을 남긴다
  # TRT_EXTRA: 메모리 부족 대응 등 추가 빌드 플래그(예: --memPoolSize=workspace:1024). 정밀도 설정은
  # 바꾸지 않는다. 쓴 값은 results/build_flags.txt에 남겨 결과와 함께 보고한다.
  echo "$(date -Is) $plan $flags ${TRT_EXTRA:-}" >> results/build_flags.txt
  # --skipInference (2026-10-02): 빌드만 하고 엔진을 저장한다. trtexec는 빌드 직후 같은 프로세스에서 시험 추론을
  # 도는데, 빌드 메모리를 쥔 채 실행 공간을 잡다가 실패해 **이미 만든 엔진까지 지워졌다**(802,816 강타입, 6.4h).
  # 측정은 orin_verify_tower.py가 따로 하므로 시험 추론은 필요 없다 — 엔진은 같다.
  "$TRTEXEC" --onnx=$m.onnx --saveEngine=$plan $flags ${TRT_EXTRA:-} --skipInference --verbose \
      --profilingVerbosity=detailed --exportLayerInfo=logs/layers_${m}_$c.json \
      > logs/build_${m}_$c.log 2>&1 \
    || { echo "  **빌드 실패** $plan — logs/build_${m}_$c.log 끝부분:"; tail -5 logs/build_${m}_$c.log; rm -f "$plan"; return 1; }
}

measure () {  # $1=모델 $2=조건
  local m=$1 c=$2 tag=${1#tower_}_$2
  [ -f results/orin_$tag.json ] && { echo "  [건너뜀] results/orin_$tag.json"; return 0; }
  [ -f ${m}_$c.plan ] || return 1
  python3 orin_verify_tower.py --engine ${m}_$c.plan --images images --list img250.txt \
      --max_pixels $MP --tag $tag --out results/orin_$tag.json 2>&1 | tee logs/measure_$tag.log
}

run () {
  if [ "${SPLIT:-0}" = 1 ]; then
    build "$1_p1" "$2" && build "$1_p2" "$2" && measure_split "$1" "$2"
  else
    build "$1" "$2" && measure "$1" "$2"
  fi
}

measure_split () {  # 두 엔진을 이어 실행 — 붕괴는 2부 최종 출력에서 판정
  local m=$1 c=$2 tag=${1#tower_}_$2
  [ -f results/orin_$tag.json ] && { echo "  [건너뜀] results/orin_$tag.json"; return 0; }
  python3 orin_verify_tower.py --engine ${m}_p1_$c.plan --engine2 ${m}_p2_$c.plan \
      --images images --list img250.txt --max_pixels $MP --tag $tag \
      --out results/orin_$tag.json 2>&1 | tee logs/measure_$tag.log
}

# ── 강타입 (6.4h) — fp16 그래프(*h)를 --stronglyTyped로. 이 모드만 돌고 끝난다 ──────
if [ "$MODE" = typed ]; then
  echo "[2] 강타입 --stronglyTyped"
  for m in $MODELS; do run $m typed; done
  exit 0
fi

# ── 2. 관행 --fp16 — 가장 중요한 셋. 붕괴 쪽부터 ────────────────────────
echo "[2] 관행 --fp16"
for m in $MODELS; do run $m fp16; done
[ "$MODE" = fp16 ] && exit 0

# ── 3. fp32 대조군 (baseline만) ─────────────────────────────────────────
echo "[3] fp32 대조군"
run $BASE fp32

# ── 4. 엄격 — A100에서도 빌드가 실패했던 조건. 실패해도 결론은 선다 ────
echo "[4] 엄격"
for m in $MODELS; do run $m strict || true; done

# ── 5. 빌드 로그에서 마지막 블록 down_proj 정밀도 발췌 (논문 6.2) ─────────
: > results/down_proj_precision.txt
for f in logs/layers_*.json; do
  [ -f "$f" ] || continue
  python3 - "$f" >> results/down_proj_precision.txt <<'PY'
import json, sys
f = sys.argv[1]; d = json.load(open(f))
layers = d.get("Layers", d) if isinstance(d, dict) else d
hit = [l for l in layers if isinstance(l, dict) and "blocks.31/mlp/down_proj" in json.dumps(l)]
print(f"== {f}  (레이어 {len(layers)}개, blocks.31 down_proj 포함 {len(hit)}개)")
for l in hit[:3]:
    ins = [(t.get("Name","")[-40:], t.get("Format/Datatype")) for t in l.get("Inputs", [])][:3]
    outs = [(t.get("Name","")[-40:], t.get("Format/Datatype")) for t in l.get("Outputs", [])][:3]
    print("  ", l.get("Name","")[:80], "| in", ins, "| out", outs)
PY
done
echo "[5] 발췌 → results/down_proj_precision.txt"

tar czf orin_results_${RES}_$(date +%Y%m%d_%H%M).tar.gz results logs
echo "끝. orin_results_*.tar.gz 를 서버로 가져올 것"
