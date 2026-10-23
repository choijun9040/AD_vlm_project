#!/bin/bash
# 평가 데이터 전부로 통일 — 4단계: TensorRT (A100) 6장 수치 전부를 799장으로 (사전 등록 4fddeb4, 실행 순서 5)
#
# DriveLM 검증 CAM_FRONT 799장(149장면, build_token_to_images 표준 순서), 결과는 eval_results/full799/ 새 파일로,
# 파일 이름은 원래 결과 파일과 같게 둔다(250/50이 이름에 박힌 것만 799로 바꿨다). 이미지별 기록(--per_image)을 남긴다.
# 교정(fixA) 그래프는 **799장으로 새로 푼 가중치**(baseline_v2_fixa_native_tower_full799.pt)에서 모두 다시 내보낸다
# (옛 그래프는 100장 가중치 baseline_v2_fixa_native_tower.pt에서 나왔다 — 섞지 말 것).
#
# 해상도마다 [입력 npz 준비 → 그래프 하나 내보내기 → 그 그래프의 측정 → 그래프 삭제] 를 차례로 하고, 해상도가 끝나면 npz를
# 지운다(799장 원본 해상도 npz가 fp32로 약 27 GB). Orin 꾸러미에 들어갈 그래프(802816h·401408h × 세 모델)만 남긴다.
# 지연을 재므로 GPU가 비어 있을 때만 측정한다(맨 위에서 1~3단계가 끝나기를 기다리고, 측정마다 다시 확인한다).
#
# 다시 실행하면 이미 있는 결과·그래프·npz는 건너뛴다 — 중간에 끊겨도 그대로 다시 돌리면 된다.
#   nohup bash scripts/run_full799_s4_trt.sh > logs/run_full799_s4_trt.log 2>&1 & echo $! > logs/run_full799_s4_trt.pid
# 환경 변수:
#   TRT1016=0     TensorRT 10.16 선택 블록을 건너뛴다 (기본 1 — 아래 「선택 블록」 참조)
#   PAIR1013=0    교정 그래프 fp16·bf16 같은 세션 짝(10.13) 추가 측정을 건너뛴다 (기본 1)
#   KEEP_ALL=1    측정이 끝난 그래프를 지우지 않는다 (디스크가 넉넉할 때)
#   ORIN_PKG=0    끝에 Orin 꾸러미(make_orin_pkg_full799.sh)를 만들지 않는다 (기본 1)
set -u
cd /workspace/vlm_project_2; export HF_HOME=/workspace/hf_home
PY=/workspace/venv_main/bin/python          # 전처리·내보내기 (torch 2.3, transformers 4.49)
PT=/workspace/venv_trt/bin/python           # TensorRT 10.13.3.9
PT2=/workspace/venv_trt2/bin/python         # TensorRT 10.16.1.11 (선택 블록에서 만든다)
O=eval_results/full799; G=onnx/full799; L=logs/full799_trt; mkdir -p $O $G $L
FIX=checkpoints/baseline_v2_fixa_native_tower_full799.pt
BASE=checkpoints/student_baseline_v2/epoch_1; FULL=checkpoints/student_full/epoch_1
N=799
BUSY="full799_queue6?\.sh|run_full799_s[123]r?\.sh|run_full799_s4a\.sh|headroom_guard.py fix"   # 2026-10-22: s4a·6판 대기열 추가
log(){ echo "[$(date '+%F %T')] $*"; }

# ── 0. 기다림 — 799장 교정 가중치가 다 써지고, 1~3단계 GPU 작업이 끝날 때까지 ─────────────────────
log "대기: $FIX 와 GPU 작업($BUSY) 종료"
until [ -f $FIX ] && ! pgrep -f "headroom_guard.py fix" >/dev/null; do sleep 60; done
# 이어 하기 판(2026-10-21): 1~3단계의 **완료 표시**를 기다린다 — 프로세스 이름만 보면 단계 사이의 빈틈에 일찍 시작할 수 있다
until [ -f eval_results/full799/.stage_queue_done ]; do sleep 120; done   # 대기열(1~3단계 작업 100개)이 끝난 뒤
until [ -f eval_results/full799/.stage_s4a_done ]; do sleep 120; done     # 앞 단계(s4a, 지연을 쓰지 않는 측정)가 끝난 뒤 — 2026-10-22
while pgrep -f "$BUSY" >/dev/null; do sleep 60; done
sleep 60   # 마지막 파일 쓰기가 끝나도록
log "교정 가중치 $(ls -l --time-style=+%F_%T $FIX | awk '{print $5, $6}') sha256 $(sha256sum $FIX | cut -c1-16)"
$PT -c "import tensorrt as t; print('[TensorRT]', t.__version__)" || { log "**venv_trt에 tensorrt가 없다 — 중단**"; exit 1; }
$PT -c "import tensorrt as t, sys; sys.exit(0 if t.__version__ == '10.13.3.9' else 1)" \
  || log "**주의: venv_trt의 TensorRT가 10.13.3.9가 아니다 — 원래 측정과 버전이 다르다. 결과 JSON의 trt 필드로 확인할 것**"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/[GPU] /'

# GPU가 비었는지 — 지연 측정 직전마다 부른다 (다른 프로세스가 돌면 지연이 섞인다)
wait_idle(){
  local said=0
  # MPS 서버(상주, 약 32 MiB)는 늘 목록에 있으므로 64 MiB 이하 항목은 뺀다 (2026-10-23 — 빼지 않으면 영원히 기다린다)
  while pgrep -f "$BUSY" >/dev/null || [ -n "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '$1 > 64')" ]; do
    [ $said = 0 ] && { log "  GPU 사용 중 — 빌 때까지 기다린다"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; said=1; }
    sleep 60
  done
}
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
  wait_idle
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

# ══ 1. 원본 해상도 1,440,000 (7,296 패치) ═══════════════════════════════════════════════
# 6.3 [trt-result]·6.4 [four-options]·1.34~1.38배·6.5 [typed-a100] 원본 행
NPZ=$G/trt_inputs_native799.npz
if prep 1440000 $NPZ 36; then
  # (a) 무교정 단일 그래프 (tower_baseline_v2_native) — fp32 대조군·fp16 관행·엄격·bf16
  if exp tower_baseline_v2_native 1440000 $BASE; then
    probe trt_precision_probe         $NPZ fp32,fp16,strict tower_baseline_v2_native   # 원래: trt_precision_probe.json (50장)
    probe trt_precision_bf16          $NPZ fp32,fp16,bf16   tower_baseline_v2_native   # 원래: trt_precision_bf16.json (50장)
    probe trt_native799_baseline_v2_dense $NPZ fp16         tower_baseline_v2_native   # 원래: trt_native250_baseline_v2_dense.json (250장, 92.8%)
    drop tower_baseline_v2_native
  fi
  # (b) Full 단일 그래프 — 원래 TRT 결과 파일 없음(새로 잰다: [typed-a100] "Full과 교정본은 모든 칸에서 0%"의 관행 fp32 그래프 칸)
  if exp tower_full_native 1440000 $FULL; then
    probe trt_native799_full_dense $NPZ fp16 tower_full_native
    drop tower_full_native
  fi
  # (c) 교정 단일 그래프 — 799장 가중치. 교정 fp16(149.5 ms)·bf16(205.5 ms) → 1.375배, 10.16에서 1.344배
  if exp tower_baseline_v2_native_fixa 1440000 $BASE --vision_state $FIX; then
    probe trt_fixa_native          $NPZ fp16 tower_baseline_v2_native_fixa      # 원래: trt_fixa_native.json (10.13, fp16만)
    probe trt_fixa_native_bf16_1013 $NPZ bf16 tower_baseline_v2_native_fixa     # 원래: trt_fixa_native_bf16_1013.json (10.13, bf16만)
    # 추가: 같은 세션 짝 (10.13). 원래 10.13의 두 값은 **다른 날·다른 실행**(09-16 10:15 / 17:09)이었는데 6장 본문은
    # 「같은 그래프·같은 세션」이라고 적는다. 10.16(아래)처럼 한 실행에서 fp16,bf16을 함께 재 그 서술을 확인한다.
    [ "${PAIR1013:-1}" = 1 ] && probe trt_fixa_native_pair_1013 $NPZ fp16,bf16 tower_baseline_v2_native_fixa
    # ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
    # │ 선택 블록 — TensorRT 10.16 (원래 /venv/trt2, 지워졌다). 6.4의 「버전을 올려도 1.344배」 하나에만 쓰인다. │
    # │ /workspace/venv_trt2를 만들고 교정 그래프의 fp16,bf16 지연 비교만 다시 잰다. TRT1016=0이면 건너뛴다.     │
    # └──────────────────────────────────────────────────────────────────────────────────────────────┘
    if [ "${TRT1016:-1}" = 1 ]; then
      if [ ! -x $PT2 ] || ! $PT2 -c "import tensorrt" 2>/dev/null; then
        need_gb 6
        log "시작 [선택] venv_trt2 만들기 (tensorrt-cu12==10.16.1.11)"
        # cuda-python은 venv_trt와 같은 판으로 맞춘다(드라이버 호환 — 측정 스크립트는 두 import 경로를 다 받는다)
        CP=$($PT -c "import importlib.metadata as m; print(m.version('cuda-python'))" 2>/dev/null)
        { python3 -m venv /workspace/venv_trt2 && $PT2 -m pip install -q --upgrade pip \
          && $PT2 -m pip install -q tensorrt-cu12==10.16.1.11 "cuda-python${CP:+==$CP}" ml_dtypes numpy onnx; } > $L/venv_trt2.log 2>&1 \
          || log "**[선택] venv_trt2 설치 실패 — 로그 $L/venv_trt2.log. 10.16 측정을 건너뛴다**"
      fi
      if $PT2 -c "import tensorrt as t, sys; print('[TensorRT 10.16 블록]', t.__version__); sys.exit(0 if t.__version__.startswith('10.16') else 1)" 2>/dev/null; then
        PTX=$PT2 probe trt_fixa_native_1016 $NPZ fp16,bf16 tower_baseline_v2_native_fixa   # 원래: trt_fixa_native_1016.json (50장)
      else
        log "**[선택] venv_trt2의 TensorRT가 10.16이 아니거나 없다 — trt_fixa_native_1016 건너뜀**"
      fi
    fi
    drop tower_baseline_v2_native_fixa
  fi
  # (d) fp16 그래프(분할 16 + 나눠 계산) — 강타입([typed-a100] 강타입 열)·관행 --fp16([typed-a100] fp16 그래프 열)
  #     원본 해상도는 Orin에서 빌드 불가였으므로(6.5) 보드로 보내지 않는다 — 측정 뒤 지운다
  hset nativeh 1440000 $NPZ 0
  rm -f $NPZ && log "입력 삭제 $NPZ ($(df -h /workspace | tail -1 | awk '{print "여유 " $4}'))"
fi

# ══ 2. 802,816 (3,864 패치) ═══════════════════════════════════════════════════════════
# [typed-a100]·[orin-result]의 802,816 행. *h 그래프는 Orin 꾸러미용으로 남긴다
NPZ=$G/trt_inputs_802816_799.npz
if prep 802816 $NPZ 20; then
  if exp tower_baseline_v2_802816 802816 $BASE; then
    probe trt_802816_baseline_v2 $NPZ fp32,fp16 tower_baseline_v2_802816     # 원래: trt_802816_baseline_v2.json (관행 fp32 그래프 4.0%)
    drop tower_baseline_v2_802816
  fi
  if exp tower_full_802816 802816 $FULL; then
    probe trt_802816_full $NPZ fp16 tower_full_802816                        # 원래: trt_802816_full.json
    drop tower_full_802816
  fi
  if exp tower_baseline_v2_802816_fixa 802816 $BASE --vision_state $FIX; then
    probe trt_802816_fixa $NPZ fp16 tower_baseline_v2_802816_fixa            # 원래: trt_802816_fixa.json
    drop tower_baseline_v2_802816_fixa
  fi
  hset 802816h 802816 $NPZ 1     # 남긴다 — Orin 꾸러미 (A100 강타입 값이 보드 비교 기준)
  rm -f $NPZ && log "입력 삭제 $NPZ"
fi

# ══ 3. 401,408 (1,920 패치) ═══════════════════════════════════════════════════════════
NPZ=$G/trt_inputs_401408_799.npz
if prep 401408 $NPZ 12; then
  if exp tower_baseline_v2_401408_single 401408 $BASE; then
    probe trt_401408_baseline_v2_single $NPZ fp16,fp32 tower_baseline_v2_401408_single   # 원래: 같은 이름 (관행 fp32 그래프 15.2%)
    drop tower_baseline_v2_401408_single
  fi
  # 원래 401,408의 Full·교정본 관행 결과는 분할 그래프(trt_401408s_*)뿐이었다 — 단일 그래프로 새로 잰다
  if exp tower_full_401408_single 401408 $FULL; then
    probe trt_401408_full_single $NPZ fp16 tower_full_401408_single
    drop tower_full_401408_single
  fi
  if exp tower_baseline_v2_401408_single_fixa 401408 $BASE --vision_state $FIX; then
    probe trt_401408_fixa_single $NPZ fp16 tower_baseline_v2_401408_single_fixa
    drop tower_baseline_v2_401408_single_fixa
  fi
  hset 401408h 401408 $NPZ 1     # 남긴다 — Orin 꾸러미
  rm -f $NPZ && log "입력 삭제 $NPZ"
fi

# ── 요약 — 결과 파일별 상실률·지연 (판정은 하지 않는다: 사전 등록 4fddeb4의 기준표로 사람이 본다) ──────────
$PY - <<'PY'
import json, glob
for f in sorted(glob.glob("eval_results/full799/trt_*.json")):
    d = json.load(open(f))
    for m, r in d["modes"].items():
        if "nan_rate" in r:
            lat = r.get("latency_ms") or {}
            print(f"  {f.split('/')[-1]:42} {m:6} TRT {d['trt']:10} n={d['n_images']} 상실 {r['nan_rate']*100:5.1f}% "
                  f"({round(r['nan_rate']*d['n_images'])}장)  지연 p50 {lat.get('p50', float('nan')):7.1f} ms  가중치 {r['weights_mib'] or 0:,.0f} MiB")
        else:
            print(f"  {f.split('/')[-1]:42} {m:6} 실패 — {r.get('failed', '')[:60]}")
PY
ls -la $G | grep -E "\.onnx" | awk '{s+=$5} END {printf "[남은 그래프] %.1f GB\n", s/2^30}'

# ── Orin 꾸러미 (802816h·401408h × 세 모델 그래프 + 799장) ──────────────────────────────
[ "${ORIN_PKG:-1}" = 1 ] && bash scripts/make_orin_pkg_full799.sh
log "4단계 완료"
# 감시 스크립트가 끝을 알 수 있도록 완료 표시를 남긴다
touch eval_results/full799/.stage_s4q_done; echo "[$(date '+%F %T')] s4q 단계 완료"
