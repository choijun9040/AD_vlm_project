#!/bin/bash
# 평가 데이터 전부로 통일 — 4단계 앞 단계(s4a): TensorRT 측정 중 **지연을 쓰지 않는 것**을 대기열과 함께 미리 한다 (2026-10-22, 사용자 결정)
#
# 4단계(run_full799_s4q.sh)는 대기열이 다 끝난 뒤 GPU를 혼자 쓸 때만 돌도록 되어 있어 6~10시간이 맨 뒤에 그대로 붙었다.
# trt_precision_probe.py는 상실률과 지연을 같은 반복에서 재므로, GPU를 나눠 쓰면 **지연만** 흔들리고 상실률은 같다.
# 그래서 논문이 지연을 쓰지 않는 측정을 여기서 먼저 한다:
#   401,408·802,816 전부(단일 그래프 관행·fp32, 분할 그래프 강타입·관행), 원본 해상도의 Full 단일(dense)·분할 그래프(nativeh).
# 지연을 쓰는 원본 해상도 단일 그래프 — tower_baseline_v2_native(trt_precision_probe·bf16·dense)와 교정 그래프
# (trt_fixa_native*, 1.375·1.344배) — 은 **하지 않는다.** s4q가 대기열과 이 단계가 끝난 뒤 GPU를 혼자 쓸 때 잰다
# (s4q는 이미 있는 결과를 건너뛰므로 여기서 끝낸 것은 다시 재지 않는다).
# 여기서 만든 결과 JSON에는 latency_gpu_shared=true를 남긴다 — 그 latency_ms는 논문에 쓰지 않는다.
# GPU 메모리는 대기열(full799_queue6.sh)과 **같은 예약 장부**(eval_results/full799/.lock_<이름>/{pid,gb})에 잡는다 —
# 대기열의 예약 합에 들어가므로 서로 메모리를 넘치게 쓰지 않는다. 잡을 때 대기열의 시작 판단 잠금(.start.lock)을 잠깐만 쥔다.
# 802,816·401,408이 끝나면 Orin 꾸러미를 만든다(보드 측정을 A100 종료 전에 시작할 수 있게).
set -u
cd /workspace/vlm_project_2; export HF_HOME=/workspace/hf_home
PY=/workspace/venv_main/bin/python          # 전처리·내보내기 (torch 2.3, transformers 4.49)
PT=/workspace/venv_trt/bin/python           # TensorRT 10.13.3.9
PT2=/workspace/venv_trt2/bin/python         # TensorRT 10.16.1.11 (선택 블록에서 만든다)
O=eval_results/full799; G=onnx/full799; L=logs/full799_trt; mkdir -p $O $G $L
FIX=checkpoints/baseline_v2_fixa_native_tower_full799.pt
BASE=checkpoints/student_baseline_v2/epoch_1; FULL=checkpoints/student_full/epoch_1
N=799
BUSY="full799_queue.sh|run_full799_s[123]r?\.sh|headroom_guard.py fix"
log(){ echo "[$(date '+%F %T')] $*"; }

BUDGET=${BUDGET:-80}
gpu_free_gb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }
alive(){ [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }
reserved_gb(){ local t=0 d p; for d in $O/.lock_* $O/.resv_*; do [ -d "$d" ] || continue
  p=$(cat "$d/pid" 2>/dev/null); alive "$p" && t=$((t + $(cat "$d/gb" 2>/dev/null || echo 0))); done; echo $t; }
RES_DIR=$O/.lock_s4a_trt
# reserve <GB> — 예약 장부에 자리가 날 때까지 기다린다. 기다리는 동안 시작 판단 잠금은 쥐지 않는다(대기열을 막지 않는다)
reserve(){
  local need=$1 said=0
  while true; do
    exec 8>"$O/.start.lock"; flock 8
    if [ $(( $(reserved_gb) + need )) -le $BUDGET ] && [ "$(gpu_free_gb)" -ge "$need" ]; then
      rm -rf $RES_DIR; mkdir -p $RES_DIR; echo $$ > $RES_DIR/pid; echo $need > $RES_DIR/gb
      flock -u 8; exec 8>&-; log "예약 ${need} GB (예약 합 $(reserved_gb)/$BUDGET GB, 실제 여유 $(gpu_free_gb) GB)"; return 0
    fi
    flock -u 8; exec 8>&-
    [ $said = 0 ] && { log "예약 ${need} GB 자리 없음 — 기다린다 (예약 합 $(reserved_gb)/$BUDGET GB)"; said=1; }
    sleep 60
  done
}
release(){ rm -rf $RES_DIR; log "예약 해제"; }
trap release EXIT
until [ -f $FIX ]; do log "799장 교정 가중치 $FIX 기다림"; sleep 300; done
# 디스크 — 다운로드 묶음(작업마다 15~40 GB)과 다투지 않도록 ${S4A_DISK:-60} GB 이상 빈 뒤에 시작한다(Gemma 종단 뒤 47 GB가 열린다)
said=0; while [ "$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)" -lt ${S4A_DISK:-60} ]; do
  [ $said = 0 ] && { log "디스크 ${S4A_DISK:-60} GB 이상 빌 때까지 기다린다 ($(df -h /workspace | tail -1 | awk '{print "여유 " $4}'))"; said=1; }; sleep 300; done
log "교정 가중치 $(ls -l --time-style=+%F_%T $FIX | awk '{print $5, $6}') sha256 $(sha256sum $FIX | cut -c1-16)"
$PT -c "import tensorrt as t; print('[TensorRT]', t.__version__)" || { log "**venv_trt에 tensorrt가 없다 — 중단**"; exit 1; }

# 디스크 — 필요한 만큼 빌 때까지 기다린다 (디스크는 나중에 늘어난다)
need_gb(){
  local said=0
  while [ "$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)" -lt "$1" ]; do
    [ $said = 0 ] && { log "  디스크 부족 — ${1} GB가 빌 때까지 기다린다 ($(df -h /workspace | tail -1))"; said=1; }
    sleep 300
  done
}

# ── 내보내기 ─────────────────────────────────────────────────────────────────────────
# exp <이름> <max_pixels> <체크포인트> [추가 인자…]  → $G/<이름>.onnx (분할이면 <이름>_p1/_p2.onnx)
# 원래 그래프와 같은 설정: 단일(dense, fp32) 그래프는 기본값, *h 그래프는 --split 16 --attn chunked --dtype fp16(6.4g).
# min_pixels는 측정 입력(prep_trt_inputs.py)과 같은 3136 — 1600×900 이미지에서는 세 해상도 모두 하한이 걸리지 않아
# 원래 내보내기(기본값 50176)와 형상이 같다.
exp(){
  local name=$1 mp=$2 ck=$3; shift 3
  if [ -f $G/$name.onnx ] || [ -f $G/${name}_p2.onnx.data ]; then log "  [건너뜀] 그래프 $name 있음"; return 0; fi
  need_gb 8
  log "시작 내보내기 $name (max_pixels $mp)"
  $PY scripts/export_vision_tower_onnx.py --checkpoint $ck --max_pixels $mp --min_pixels 3136 --out $G/$name.onnx "$@" \
      > $L/export_$name.log 2>&1
  local rc=$?
  grep -E "^\[|완료|일치|통과|경고|Error|Traceback|중단" $L/export_$name.log | tail -14
  if [ $rc -ne 0 ] || ! { [ -f $G/$name.onnx ] || [ -f $G/${name}_p2.onnx.data ]; }; then
    log "**내보내기 실패 $name (종료 $rc) — 이 그래프의 측정을 건너뛴다. 로그 $L/export_$name.log**"; return 1
  fi
  log "끝 내보내기 $name"
}
# 그래프 삭제 (Orin 꾸러미용이 아니면)
drop(){
  [ "${KEEP_ALL:-0}" = 1 ] && return 0
  rm -f $G/$1.onnx $G/$1.onnx.data $G/${1}_p1.onnx $G/${1}_p1.onnx.data $G/${1}_p2.onnx $G/${1}_p2.onnx.data
  log "  그래프 삭제 $1 ($(df -h /workspace | tail -1 | awk '{print "여유 " $4}'))"
}

# ── 측정 ─────────────────────────────────────────────────────────────────────────────
# probe <결과 이름> <npz> <조건들> <그래프 이름> [split]   (PTX=다른 venv 파이썬으로 바꿀 수 있다)
# 결과: $O/<결과 이름>.json, 이미지별 $O/<결과 이름>_per_image.jsonl, 빌드 로그 $L/<결과 이름>_build_<조건>.log
# (JSON의 build_log 필드는 logs/trt_build_<조건>.log로 남는다 — 측정 직후 위 경로로 옮겨 다음 측정에 덮이지 않게 한다)
probe(){
  local name=$1 npz=$2 modes=$3 g=$4 split=${5:-} py=${PTX:-$PT}
  local out=$O/$name.json
  [ -f $out ] && { log "  [건너뜀] $out 있음"; return 0; }
  local a=(--onnx $G/$g.onnx)
  [ "$split" = split ] && a=(--onnx $G/${g}_p1.onnx --onnx2 $G/${g}_p2.onnx)
  for f in "${a[@]}"; do case $f in *.onnx) [ -f $f ] || { log "**그래프 없음 $f — $name 건너뜀**"; return 1; };; esac; done
  log "시작 $name ($modes, $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1))"
  for m in ${modes//,/ }; do rm -f logs/trt_build_$m.log logs/trt_build_${m}_p2.log; done
  $py scripts/trt_precision_probe.py "${a[@]}" --npz $npz --modes $modes --out $out --per_image $O/${name}_per_image.jsonl \
      > $L/$name.log 2>&1
  local rc=$?
  grep -E "^\[입력\]|상실|지연|실패|저장|이미지별|Error|Traceback" $L/$name.log | tail -14
  for m in ${modes//,/ }; do
    [ -f logs/trt_build_$m.log ] && mv logs/trt_build_$m.log $L/${name}_build_$m.log
    [ -f logs/trt_build_${m}_p2.log ] && mv logs/trt_build_${m}_p2.log $L/${name}_build_${m}_p2.log
  done
  [ -f $out ] && $PY - $out <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); d["latency_gpu_shared"] = True
d["latency_note"] = "s4a: 대기열 측정과 GPU를 나눠 쓴 상태에서 측정 — 이 파일의 latency_ms는 쓰지 않는다(상실률·가중치 메모리만 쓴다)"
open(p, "w").write(json.dumps(d, ensure_ascii=False, indent=2))
PY
  log "끝 $name (종료 $rc)"
}

# ── 입력 npz ─────────────────────────────────────────────────────────────────────────
# prep <max_pixels> <npz> <필요 GB>
prep(){
  local mp=$1 npz=$2 gb=$3
  if [ ! -f $npz ]; then
    need_gb $gb
    log "시작 입력 준비 $npz (max_pixels $mp, ${N}장)"
    $PY scripts/prep_trt_inputs.py --limit $N --offset 0 --max_pixels $mp --min_pixels 3136 --out $npz.tmp.npz > $L/prep_$mp.log 2>&1 \
      && mv $npz.tmp.npz $npz
    grep -E "^\[|형상|저장|주의|건너뜀|Error|Traceback" $L/prep_$mp.log | tail -6
  fi
  [ -f $npz ] || { log "**입력 준비 실패 $npz — 이 해상도를 건너뛴다**"; return 1; }
  # 799장 모두 들어갔는지 (형상이 다른 이미지는 prep이 뺀다 — 1600×900만 있으므로 없어야 정상)
  $PY - $npz $N <<'PY' || log "**주의: npz 장 수가 ${N}이 아니다 — 결과의 n_images로 확인하고 표본 수를 명시할 것**"
import sys, numpy as np
z = np.load(sys.argv[1]); k = [x for x in z.files if not x.startswith("_")]
print(f"  [npz] {len(k)}장, 이름 {len(z['_names'])}개, 첫 장 {z['_names'][0]}")
sys.exit(0 if len(k) == int(sys.argv[2]) == len(z["_names"]) else 1)
PY
}

# fp16 그래프(*h: 블록 16 분할 + 어텐션 나눠 계산 + 가중치·활성 fp16, 6.4g) 세 모델 — 강타입(6.4h)과 관행 --fp16
# hset <태그 nativeh|802816h|401408h> <max_pixels> <npz> <남길지 0|1>
HARGS="--split 16 --attn chunked --dtype fp16"
hset(){
  local tag=$1 mp=$2 npz=$3 keep=$4 M g
  for M in baseline_v2 full fixa; do
    case $M in
      baseline_v2) g=tower_baseline_v2_$tag;        exp $g $mp $BASE $HARGS ;;
      full)        g=tower_full_$tag;               exp $g $mp $FULL $HARGS ;;
      fixa)        g=tower_baseline_v2_${tag}_fixa; exp $g $mp $BASE --vision_state $FIX $HARGS ;;
    esac || continue
    probe trt_${tag}_typed_$M $npz typed $g split   # 원래: 같은 이름 (250장, TRT 10.13 강타입)
    probe trt_${tag}_$M       $npz fp16  $g split   # 원래: 같은 이름 (250장, TRT 10.13 관행 --fp16)
    [ "$keep" = 1 ] || drop $g
  done
}


# ══ 401,408 (1,920 패치) ═══════════════════════════════════════════════════════════
NPZ=$G/trt_inputs_401408_799.npz
reserve ${GB401:-32}
if prep 401408 $NPZ 12; then
  if exp tower_baseline_v2_401408_single 401408 $BASE; then
    probe trt_401408_baseline_v2_single $NPZ fp16,fp32 tower_baseline_v2_401408_single
    drop tower_baseline_v2_401408_single
  fi
  if exp tower_full_401408_single 401408 $FULL; then
    probe trt_401408_full_single $NPZ fp16 tower_full_401408_single
    drop tower_full_401408_single
  fi
  if exp tower_baseline_v2_401408_single_fixa 401408 $BASE --vision_state $FIX; then
    probe trt_401408_fixa_single $NPZ fp16 tower_baseline_v2_401408_single_fixa
    drop tower_baseline_v2_401408_single_fixa
  fi
  hset 401408h 401408 $NPZ 1
  rm -f $NPZ && log "입력 삭제 $NPZ"
fi
release

# ══ 802,816 (3,864 패치) ═══════════════════════════════════════════════════════════
NPZ=$G/trt_inputs_802816_799.npz
reserve ${GB802:-40}
if prep 802816 $NPZ 20; then
  if exp tower_baseline_v2_802816 802816 $BASE; then
    probe trt_802816_baseline_v2 $NPZ fp32,fp16 tower_baseline_v2_802816
    drop tower_baseline_v2_802816
  fi
  if exp tower_full_802816 802816 $FULL; then
    probe trt_802816_full $NPZ fp16 tower_full_802816
    drop tower_full_802816
  fi
  if exp tower_baseline_v2_802816_fixa 802816 $BASE --vision_state $FIX; then
    probe trt_802816_fixa $NPZ fp16 tower_baseline_v2_802816_fixa
    drop tower_baseline_v2_802816_fixa
  fi
  hset 802816h 802816 $NPZ 1
  rm -f $NPZ && log "입력 삭제 $NPZ"
fi
release

# ── Orin 꾸러미 — 802816h·401408h 그래프와 그 A100 강타입 결과가 있으면 만든다 ──────────────────────
if [ "${ORIN_PKG:-1}" = 1 ]; then
  bash scripts/make_orin_pkg_full799.sh && touch $O/.stage_orin_pkg_done && log "Orin 꾸러미 완료"
fi

# ══ 원본 해상도 — 지연을 쓰지 않는 것만: Full 단일 그래프(dense)와 분할 그래프(nativeh) ═════════════════
NPZ=$G/trt_inputs_native799.npz
reserve ${GBNAT:-48}
if prep 1440000 $NPZ 36; then
  if exp tower_full_native 1440000 $FULL; then
    probe trt_native799_full_dense $NPZ fp16 tower_full_native
    drop tower_full_native
  fi
  hset nativeh 1440000 $NPZ 0
  # npz는 지우지 않는다 — s4q가 원본 해상도 단일 그래프(지연)를 잴 때 다시 쓴다
fi
release
log "4단계 앞 단계(s4a) 완료"
touch $O/.stage_s4a_done; echo "[$(date '+%F %T')] s4a 단계 완료"
