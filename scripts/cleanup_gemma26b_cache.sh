#!/bin/bash
# Gemma 4 종단 측정(A 종단 799)이 끝난 뒤 google/gemma-4-26B-A4B-it 전체 가중치를 hf 캐시에서 지운다 (2026-10-22).
# 이 캐시는 모델 폴더의 blobs/<해시>가 hub/blobs/<두 글자>/<해시>(약 47 GB)를 가리키는 꼴이라, 모델 폴더만 지우면 실제 파일이 남는다.
# 링크가 가리키는 대상을 먼저 지우고 모델 폴더를 지운다. 종단 결과 JSON이 있을 때만 지운다.
set -eu
H=${HF_HOME:-/workspace/hf_home}/hub; M=$H/models--google--gemma-4-26B-A4B-it
[ -s /workspace/vlm_project_2/eval_results/full799/gemma4_e2e_26B-A4B_full799.json ] || { echo "종단 결과가 없다 — 지우지 않는다"; exit 0; }
[ -d "$M" ] || { echo "캐시 없음 — 할 일 없음"; exit 0; }
before=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
find "$M/blobs" -type l | while read -r l; do tgt=$(readlink -f "$l"); case "$tgt" in "$H"/*) rm -f "$tgt";; esac; done
rm -rf "$M"
find "$H/blobs" -mindepth 1 -type d -empty -delete 2>/dev/null || true
echo "Gemma 26B 캐시 삭제 — 디스크 여유 ${before} GB → $(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9) GB"
