# Orin Nano 측정 — 공식 Qwen2.5-VL-3B, DriveLM val CAM_FRONT 799장 (공식 처리기 기본값, 896×1596)

사전 등록 `thesis_outline_20260910.md` 개정 30(2026-11-04). 공식 `Qwen/Qwen2.5-VL-3B-Instruct` 비전 타워를 fp16 엔진으로 빌드해
DriveLM val CAM_FRONT **799장 전부**(1600×900 → 공식 처리기 기본값에서 896×1596, 패치 7,296 — 한 형상)에서 잰다.
조건은 DriveLM 799장 측정과 같다 — 강타입 fp16 그래프(블록 16 분할 + 어텐션 나눠 계산), Python 빌더, 분할 엔진 순차 측정, 15W.

| 그래프 | 내용 |
|---|---|
| `tower_official_dlm1596h` | 공식 모델 무교정 |
| `tower_official_dlm1596h_resid` | 공식 모델 (R) 교정본 — `official_residual_dlm.pt`(c는 DriveLM train 100장으로 정함), eps는 그래프 상수 |

엔진은 입력 형상마다 따로 만든다(내보낸 그래프에 격자에 따른 위치 부호·창 색인·구간 경계가 상수로 들어가 있다). DriveLM은 형상이 하나라 엔진 하나로 799장 전부를 잰다.

## 옮기기 (서버 → 보드)

```
orin_pkg_dlm_<날짜>.tar.gz   스크립트 · 이미지 799장(JPG) · 전처리 캐시 · 체크섬
orin_onnx_dlm_<날짜>.tar     ONNX 4조각(+ .data)
SHA256SUMS_dlm
```

서버의 `/dev/shm/dlm799/`에 있다(램 디스크 — 서버를 다시 켜면 사라진다). 보드의 같은 디렉터리에 두 묶음을 풀면 `orin_pkg/` 하나로 들어간다.
`sha256sum -c SHA256SUMS_dlm`로 먼저 확인한다.

## 실행 (보드)

원본 해상도 크기의 엔진은 데스크톱을 켠 채로는 빌드 메모리가 모자란다(2026-11-04 DriveLM 원본 해상도 기록). 먼저 데스크톱을 끈다.

```
sudo systemctl isolate multi-user.target
free -m                      # available 6,000 MiB 넘게
cd orin_pkg
bash orin_run_dlm.sh
```

다시 실행하면 이미 만든 엔진·결과는 건너뛴다. 끝나면 `orin_results_dlm_<시각>.tar.gz`와 `.sha256`이 생긴다 — 서버로 돌려보낸다.
마지막에 A100 fp16 엔진 값(`a100_ref/trt_dlm1596h_typed_*.json`)과 나란히 출력한다. 데스크톱은 `sudo systemctl isolate graphical.target`으로 되돌린다.
