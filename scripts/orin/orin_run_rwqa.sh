#!/usr/bin/env bash
# Orin Nano 실측 — 공식 Qwen2.5-VL-3B 비전 타워, RealWorldQA 원본 해상도 952×1456 형상 465문항 (사전 등록 개정 23, 2026-11-04)
#
# orin_run_full799.sh와 같은 절차(Python 빌더 fp16 엔진, 블록 16 분할 + 어텐션 나눠 계산, 두 분할 엔진을 다른 프로세스에서 차례로,
# 페이지 캐시를 비운 상태, 데스크톱을 끈 상태 — `sudo systemctl isolate multi-user.target`). 바뀐 것: 이미지(RealWorldQA PNG 465장,
# img465.txt), 형상(공식 처리기 기본값 max_pixels 12,845,056 → 952×1456, 패치 7,072), 모델(공식 무교정·(R) 교정본).
#
#   bash orin_run_rwqa.sh
# 다시 실행하면 이미 만든 엔진·결과는 건너뛴다 — 중간에 끊겨도 그대로 다시 돌리면 된다.
set -u
cd "$(dirname "$0")"
RESL=${*:-rwqa952h}
LIST=img465.txt
export HF_HOME=$PWD/hf_cache_processor HF_HUB_OFFLINE=1
mkdir -p results logs
log(){ echo "[$(date '+%F %T')] $*"; }

N=$(grep -c . $LIST)
[ "$N" = 465 ] || { echo "**$LIST 가 465줄이 아니다($N) — 꾸러미를 다시 풀 것**"; exit 1; }

# 페이지 캐시 비우기 — 빌드가 메모리 경계선에 있다(6.5절 (iv)). sudo 암호 없이 안 되면 경고만 하고 계속한다
drop_caches(){
  sync
  if sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then echo "  (페이지 캐시 비움)"
  else echo "  **경고: 페이지 캐시를 비우지 못했다(sudo 암호 필요) — README의 sudo 유지 명령을 먼저 실행할 것**"
       echo "$(date -Is) drop_caches 실패" >> results/build_flags.txt; fi
}

# ── 0. 환경 기록 ───────────────────────────────────────────────────────────────────
if [ ! -f results/env.txt ]; then
  { echo "## date";      date -Is
    echo "## jetpack";   cat /etc/nv_tegra_release 2>/dev/null
    echo "## tensorrt";  python3 -c "import tensorrt; print(tensorrt.__version__)" 2>&1
    dpkg -l 2>/dev/null | grep -iE "tensorrt|nvidia-l4t-core" | awk '{print $2, $3}'
    echo "## nvpmodel";  sudo -n nvpmodel -q 2>&1
    echo "## jetson_clocks"; sudo -n jetson_clocks --show 2>&1 | head -20
    echo "## python";    python3 -c "import transformers, numpy; print('transformers', transformers.__version__, 'numpy', numpy.__version__)" 2>&1
    echo "## mem";       free -h; swapon --show 2>/dev/null
    echo "## disk";      df -h .
  } > results/env.txt
  log "[0] 환경 기록 → results/env.txt"
fi
grep -q "15W" results/env.txt || echo "**주의: 전력 모드가 15W로 기록되지 않았다(2026-10-02 측정은 15W) — results/env.txt 확인**"

# ── 1. 무결성 · 전처리 일치 — 하나라도 실패하면 멈춘다 ─────────────────────────────────
sha256sum -c --quiet SHA256SUMS_pkg || { echo "**꾸러미 파일(이미지·스크립트) 체크섬 불일치 — 다시 옮길 것**"; exit 1; }
log "[1] 꾸러미 체크섬 통과 (이미지 465장 포함)"
python3 orin_check_preproc.py || { echo "**전처리 불일치 — 측정 중단**"; exit 1; }

build(){  # $1=ONNX 이름(확장자 없이) — 강타입 엔진 $1_typed.plan. 메모리 할당 실패에 대비해 캐시를 비우고 세 번까지
  local m=$1 plan=$1_typed.plan a
  [ -f "$plan" ] && { echo "  [건너뜀] $plan"; return 0; }
  for a in 1 2 3; do
    drop_caches
    echo "$(date -Is) $plan --stronglyTyped (orin_build_engine.py, 시도 $a)" >> results/build_flags.txt
    log "  [빌드] $plan (시도 $a) — 수 분 걸린다"
    if python3 orin_build_engine.py --onnx $m.onnx --out $plan --mode typed > logs/build_${m}_typed_a$a.log 2>&1; then
      tail -1 logs/build_${m}_typed_a$a.log; return 0
    fi
    echo "  **빌드 실패** (시도 $a) — logs/build_${m}_typed_a$a.log 끝부분:"; tail -3 logs/build_${m}_typed_a$a.log
    rm -f "$plan"
  done
  return 1
}

measure(){  # $1=모델 그래프 이름(tower_…), $2=max_pixels — 두 분할 엔진을 다른 프로세스에서 차례로
  local m=$1 mp=$2 tag=${1#tower_}_typed
  [ -f results/orin_$tag.json ] && { echo "  [건너뜀] results/orin_$tag.json"; return 0; }
  local common="--engine ${m}_p1_typed.plan --engine2 ${m}_p2_typed.plan --sequential --tmp seq_tmp_$tag
                --images images --list $LIST --max_pixels $mp --tag $tag"
  drop_caches
  log "  [측정 1부] $tag (${N}장)"
  python3 orin_verify_tower.py $common --stage 1 > logs/measure_$tag.log 2>&1
  [ $? = 0 ] || { echo "  **1부 측정 실패** — logs/measure_$tag.log"; tail -5 logs/measure_$tag.log; return 1; }
  drop_caches
  log "  [측정 2부] $tag"
  python3 orin_verify_tower.py $common --stage 2 --out results/orin_$tag.json \
      --per_image results/orin_${tag}_per_image.jsonl >> logs/measure_$tag.log 2>&1
  [ -f results/orin_$tag.json ] || { echo "  **2부 측정 실패** — 엔진은 남아 있다. 다시 실행하면 측정만 다시 한다"; tail -5 logs/measure_$tag.log; return 1; }
  grep -E "붕괴율|지연|메모리" logs/measure_$tag.log | tail -3
}

# ── 2. 강타입 — 해상도마다 세 모델 ───────────────────────────────────────────────────
for RES in $RESL; do
  case $RES in
    rwqa952h) MP=12845056 ;;   # 공식 처리기 기본값 — 이 목록의 이미지는 모두 952×1456으로 처리된다
    *) echo "해상도는 rwqa952h"; exit 1 ;;
  esac
  # 모델 목록은 MODELS_TPL로 바꿀 수 있다({RES} 자리표시) — (R) 교정본 꾸러미는 MODELS_TPL="tower_baseline_v2_{RES}_resid" (2026-11-03, 개정 10)
  # 기본값을 ${MODELS_TPL:-...} 안에 바로 쓰면 {RES}의 「}」가 확장을 닫아 버려, MODELS_TPL을 줘도 기본 목록 뒷부분이 붙는다(2026-11-04 보드에서 발견)
  DEFAULT_TPL='tower_official_{RES} tower_official_{RES}_resid'
  TPL=${MODELS_TPL:-$DEFAULT_TPL}
  MODELS=$(echo "$TPL" | sed "s/{RES}/$RES/g")
  log "[2] $RES (max_pixels $MP) 강타입"
  for m in $MODELS; do
    for p in p1 p2; do
      grep -q "  ${m}_$p.onnx\$" SHA256SUMS_onnx || { echo "**SHA256SUMS_onnx에 ${m}_$p.onnx 가 없다**"; exit 1; }
    done
  done
  NEED=$(for m in $MODELS; do for p in p1 p2; do echo "${m}_$p.onnx"; echo "${m}_$p.onnx.data"; done; done)
  for f in $NEED; do [ -f "$f" ] || { echo "**파일 없음: $f — ONNX 묶음을 이 디렉터리에 풀 것**"; exit 1; }; done
  printf '%s\n' $NEED | sed 's/^/  /;s/$/$/' | grep -f - SHA256SUMS_onnx | sha256sum -c --quiet - \
    || { echo "**ONNX 체크섬 불일치 — 파일을 다시 옮길 것**"; exit 1; }
  log "  ONNX 체크섬 통과"
  for m in $MODELS; do
    build ${m}_p1 && build ${m}_p2 && measure $m $MP || log "  **$m 실패 — 다음 모델로 넘어간다(다시 실행하면 이어서 한다)**"
  done
done

# ── 3. A100 강타입 값과 나란히 (판정은 하지 않는다 — 사전 등록 6.4h·4fddeb4의 ±5%p 기준으로 서버에서 본다) ─────
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("results/orin_*_typed.json")):
    o = json.load(open(f)); tag = o["tag"][:-len("_typed")]            # baseline_v2_802816h(_fixa) / full_802816h
    res = "rwqa952h"
    m = next((sfx for sfx in ("fixa", "resid") if tag.endswith("_" + sfx)), None) or tag.replace("_" + res, "")
    ref = f"a100_ref/trt_{res}_typed_{m}.json"
    a = json.load(open(ref))["modes"]["typed"] if os.path.exists(ref) else None
    a_s = f"A100 {a['nan_rate']*100:5.1f}%" if a else "A100 기준 없음"
    print(f"  {tag:28} Orin {o['nan_rate']*100:5.1f}% ({o['nan_count']}/{o['n']})  {a_s}   지연 p50 {o['latency_ms']['p50']:.0f} ms")
PY

STAMP=$(date +%Y%m%d_%H%M)
tar czf orin_results_rwqa_$STAMP.tar.gz results logs
sha256sum orin_results_rwqa_$STAMP.tar.gz | tee orin_results_rwqa_$STAMP.tar.gz.sha256
log "끝. orin_results_rwqa_$STAMP.tar.gz 와 .sha256 을 서버로 가져올 것"
