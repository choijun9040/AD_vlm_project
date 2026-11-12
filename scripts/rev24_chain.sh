#!/bin/bash
# 개정 24 — ① 공식 모델 래퍼 검증(fp32 dense 그래프 952×1456 → TensorRT fp32, 465장) ② 공식 모델 백본(RealWorldQA 765문항). 2026-11-04
set -u
cd /workspace/vlm_project_2
PY=/workspace/venv_main/bin/python; PT=/workspace/venv_ort/bin/python
R=/dev/shm/rwqa952; V=$R/verify; mkdir -p $V
IMG=$R/images/$(head -1 $R/img465.txt)
log(){ echo "[$(date '+%F %T')] $*"; }
if [ ! -f $V/tower_official_rwqa952h_fp32.onnx ]; then
  log "① 내보내기 fp32 dense"
  $PY scripts/export_vision_tower_onnx.py --checkpoint none --max_pixels 12845056 --min_pixels 3136 --image $IMG \
      --out $V/tower_official_rwqa952h_fp32.onnx > logs/rev24_export.log 2>&1 || { tail -5 logs/rev24_export.log; exit 1; }
  grep -hE "고정 형상|일치|오차" logs/rev24_export.log | tail -4
fi
log "① TensorRT fp32 검증 (465장)"
$PT scripts/onnx_verify799.py --runtime trt --attn default --checkpoint none --max_pixels 12845056 --min_pixels 3136 \
    --list $R/paths465.txt --limit 465 --onnx $V/tower_official_rwqa952h_fp32.onnx --plan $V/tower_official_rwqa952h_fp32.plan \
    --out eval_results/cmp/rev24_wrapper_verify.json > logs/rev24_verify.log 2>&1 || { tail -5 logs/rev24_verify.log; exit 1; }
tail -3 logs/rev24_verify.log | cut -c1-400
log "② 백본 (RealWorldQA 765문항)"
$PY scripts/probe_llm_headroom.py --checkpoint none --tag official_rwqa765 --rwqa data/realworldqa/questions.jsonl --attn default \
    --max_pixels 12845056 --min_pixels 3136 --out_dir eval_results/cmp > logs/rev24_backbone.log 2>&1 || { tail -5 logs/rev24_backbone.log; exit 1; }
grep -hE "^\[크기|^\[상실|→" logs/rev24_backbone.log
log "사슬 끝"
