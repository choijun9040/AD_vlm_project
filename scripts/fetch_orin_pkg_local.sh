#!/bin/bash
# Orin 꾸러미(DriveLM)를 A100에서 로컬로 이어받기 — 끊겨도 --partial로 잇고, 다 받으면 체크섬을 확인한다. 2026-11-05
cd /home/cj/workspace/vlm_project_2/onnx/dlm
for i in $(seq 1 200); do
  rsync -a --partial --append-verify vast:/dev/shm/dlm799/orin_pkg_dlm_20261104.tar.gz vast:/dev/shm/dlm799/orin_onnx_dlm_20261104.tar vast:/dev/shm/dlm799/SHA256SUMS_dlm . 2>/dev/null && break
  echo "$(date '+%F %T') 재시도 $i"; sleep 60
done
sha256sum -c SHA256SUMS_dlm && echo "$(date '+%F %T') 꾸러미 받기 끝 — 체크섬 통과" || echo "$(date '+%F %T') **체크섬 불일치**"
