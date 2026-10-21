# Orin Nano 재측정 — 평가 데이터 전부(799장)로 통일

사전 등록 `4fddeb4`(`thesis_outline_20260910.md` §6 「평가 데이터 전부로 통일」)의 실행 순서 6이다.
6장 [orin-result]를 **DriveLM 검증 CAM_FRONT 799장(149장면)** 으로 다시 잰다. 이전 측정(2026-10-02)은 같은 순서의 앞 250장이었다.

## 무엇을 재는가 — 이전 측정과 같은 조건, 이미지만 799장

| 해상도 | 모델 (그래프) | 조건 | 이전 결과 (250장) |
|---|---|---|---|
| 802,816 | 기준선 `tower_baseline_v2_802816h` | 강타입 `--stronglyTyped`, Python 빌더, 분할 엔진 순차 측정, 15W | 24.4% (61/250), 1,067 ms |
| 802,816 | Full `tower_full_802816h` | 같음 | 0.0%, 1,069 ms |
| 802,816 | 교정본 `tower_baseline_v2_802816h_fixa` | 같음 | 0.0%, 1,069 ms |
| 401,408 | 기준선 · Full · 교정본 (`*_401408h*`) | 같음 | 9.2% (23/250) · 0% · 0%, 504 ms |

- 그래프는 모두 fp16 그래프(블록 16 분할 + 어텐션 나눠 계산, 6.4g)를 A100에서 **새로 내보낸 것**이다.
  **교정본은 799장으로 다시 푼 가중치**(`checkpoints/baseline_v2_fixa_native_tower_full799.pt`)에서 나왔다 — 이전 그래프와 체크섬이 다르다.
- 원본 해상도(1,440,000)는 넣지 않았다. 강타입으로도 빌드가 열 번 모두 같은 바이트(0.64 GB)에서 실패했고, 이는 이미지와 무관한
  빌드 단계 결과라 799장으로 바꿔도 달라지지 않는다.
- 바뀐 것은 이미지 목록(`img799.txt`)과 **이미지별 기록**(`results/orin_<tag>_per_image.jsonl`, 장면 토큰 포함)뿐이다.
  `img799.txt`의 앞 250줄은 이전 `img250.txt`와 같다(꾸러미를 만들 때 확인했다).

## 0. 옮기기 (서버 → 보드)

서버(A100)에서 `bash scripts/make_orin_pkg_full799.sh`가 만든 것:

```
onnx/orin_pkg_full799_<날짜>.tar.gz    스크립트 · 이미지 799장 · 전처리 캐시 · 체크섬 (~120 MB)
onnx/orin_onnx_full799_<날짜>.tar      ONNX 24개 (6개 그래프 × p1/p2 × .onnx/.onnx.data, ~8.6 GB)
onnx/SHA256SUMS_full799               위 두 묶음의 체크섬
```

보드에서 (이전 꾸러미와 섞이지 않게 **새 디렉터리**에서):

```bash
mkdir -p ~/orin_full799 && cd ~/orin_full799
# 서버에서 세 파일을 이 디렉터리로 가져온다 (scp/rsync 등)
sha256sum -c SHA256SUMS_full799                      # 두 묶음 모두 OK여야 한다
tar xzf orin_pkg_full799_*.tar.gz                    # → orin_pkg/
tar xf  orin_onnx_full799_*.tar                      # → orin_pkg/ 안에 ONNX 24개
cd orin_pkg
```

(2-b) ONNX 묶음이 없으면(서버 디스크 부족으로 `ONNX_TAR=0`) 서버의 `onnx/full799/`에서 아래 24개를 `orin_pkg/`로 직접 옮긴다.
`tower_{baseline_v2,full}_{802816h,401408h}_p{1,2}.onnx(.data)`, `tower_baseline_v2_{802816h,401408h}_fixa_p{1,2}.onnx(.data)`.
보드 스크립트가 `SHA256SUMS_onnx`로 검사하므로 잘못 옮기면 측정 전에 멈춘다.

## 1. 사전 준비 (보드)

```bash
cat /etc/nv_tegra_release            # 이전 측정: R36.4.7
python3 -c "import tensorrt; print(tensorrt.__version__)"   # 이전 측정: 10.3.0
sudo nvpmodel -m 0 && sudo nvpmodel -q   # 15W — 이전 측정과 같게. "NV Power Mode: 15W" 확인
free -h; df -h .                     # 디스크는 약 40 GB 필요(아래)
```

- **jetson_clocks.** 이전 측정의 `env.txt`에는 jetson_clocks 상태가 기록되지 않았다. 이전과 같은 조건이 무엇이었는지 확실하지 않으므로
  **이번에는 켜지도 끄지도 말고 그대로 두고**, 스크립트가 `jetson_clocks --show`로 상태를 `results/env.txt`에 남긴다. 지연을 이전 값과
  비교할 때 이 점을 적는다.
- **스왑.** 이전 측정에서는 zram 6개 + `/swapfile_trt` 16 GB가 켜져 있었다(`env_swap.txt`). 그대로 두면 된다.
- **sudo 유지.** 빌드·측정 전마다 페이지 캐시를 비운다(빌드가 메모리 경계선에 있다, 6.5절 (iv)). 암호 없이 sudo가 되도록,
  측정 전에 다른 터미널(또는 같은 셸 백그라운드)에서:
  ```bash
  sudo -v && while true; do sudo -n true; sleep 60; done &
  ```
  sudo가 안 되면 스크립트는 경고만 남기고 계속한다(`results/build_flags.txt`에 기록). 빌드가 메모리 부족으로 실패하면 이것부터 본다.
- 전처리 의존성(이전 꾸러미와 같다): `transformers==4.49.0`, `pillow`, `numpy`, `cuda-python`(또는 `pycuda`).

## 2. 실행 (보드)

```bash
cd ~/orin_full799/orin_pkg
nohup bash orin_run_full799.sh > run_full799.log 2>&1 & echo $! > run_full799.pid
tail -f run_full799.log
kill -0 $(cat run_full799.pid) && echo "실행 중" || echo "종료됨"
```

- 해상도 하나만: `bash orin_run_full799.sh 802816h` (또는 `401408h`). 기본은 `802816h` → `401408h`.
- 스크립트가 하는 일: 환경 기록 → 꾸러미 체크섬(이미지 799장 포함) → 전처리 해시 점검(`orin_check_preproc.py`, 이전과 같은 기준)
  → 해상도마다 ONNX 체크섬 → 모델마다 [1부·2부 강타입 빌드(실패하면 캐시를 비우고 세 번까지) → 1부 측정(799장, 은닉 상태를
  디스크에) → 새 프로세스에서 2부 측정] → A100 강타입 값(꾸러미의 `a100_ref/`)과 나란히 출력 → 결과 묶음.
- **중간에 끊기면 그대로 다시 실행한다.** 만든 엔진(`*.plan`)과 끝난 결과(`results/orin_*_typed.json`)는 건너뛴다.
  2부에서 끊겼으면 1부부터 다시 돈다(은닉 상태 디렉터리 `seq_tmp_<tag>/`는 덮어쓴다).
- 빌드가 세 번 다 실패하면 그 모델만 건너뛰고 다음으로 간다 — 로그 `logs/build_<그래프>_typed_a{1,2,3}.log`. **실패도 결과다.**
  어디서 실패했는지(할당 바이트)를 그대로 가져온다(6.5절처럼 적는다).

### 예상 시간과 디스크

이전 측정(250장)에서 지연 p50은 802,816이 **1,067~1,069 ms/장**, 401,408이 **503~504 ms/장**이었고(두 분할 엔진 합, 15W),
빌드는 엔진 하나에 1~2분(802,816 104~132초)이었다. 799장이면:

| 해상도 | 모델당 측정 | 모델당 빌드 | 세 모델 |
|---|---|---|---|
| 802,816 | 약 14분(799 × 1.07 s) + 전처리·디스크 입출력 약 3분 | 약 4~5분 | **약 65분** |
| 401,408 | 약 7분(799 × 0.50 s) + 약 2분 | 약 2~3분 | **약 35분** |

모두 합쳐 **약 1시간 40분~2시간 30분**(빌드 재시도가 생기면 더). 디스크는 802,816 1부 은닉 상태가 799 × 19.8 MB ≈ **15.8 GB**
(2부가 읽으면서 지운다), 엔진 6쌍 약 8 GB, ONNX 8.6 GB, 묶음 8.7 GB — **약 40 GB**를 비워 둔다.

## 3. 결과 보내기 (보드 → 서버)

스크립트 끝에서 `orin_results_full799_<날짜_시각>.tar.gz`와 `.sha256`이 생긴다. 안에는 `results/`(결과 JSON 6개, 이미지별 jsonl 6개,
`env.txt`, `build_flags.txt`)와 `logs/`(빌드·측정 로그 전부)가 있다.

```bash
# 서버에서
mkdir -p eval_results/orin/full799 && cd eval_results/orin/full799
# 보드에서 두 파일을 가져온 뒤
sha256sum -c orin_results_full799_*.tar.gz.sha256
tar xzf orin_results_full799_*.tar.gz        # → results/, logs/
```

서버 쪽 비교 대상은 A100 강타입 799장 `eval_results/full799/trt_{802816h,401408h}_typed_{baseline_v2,full,fixa}.json`(이미지별
`*_per_image.jsonl`)이다. 판정은 사전 등록대로 한다 — **보드 상실률이 A100 강타입 값과 ±5%p 안이면 같다**(6.4h), 런타임·보드 재현은
**PyTorch와 ±5%p**(4fddeb4). 이미지별로 같은 장이 상실했는지는 이미지 이름으로 짝지어 본다(McNemar·장면 군집 부트스트랩).
**불리하게 나와도 해석을 바꾸지 않는다.**

## 파일

| 파일 | 내용 |
|---|---|
| `orin_run_full799.sh` | 위 절차 전체 (`scripts/orin/orin_run_full799.sh`) |
| `orin_verify_tower.py` · `orin_build_engine.py` · `orin_check_preproc.py` | 이전 꾸러미와 같은 측정·빌드·전처리 점검 (`--per_image`만 추가된 판) |
| `img799.txt` · `img799_scenes.json` · `scene_map.py` | 표준 순서 이미지 목록, 이미지→장면 표, 그 표를 읽는 보드용 장면 매핑 |
| `images/` | CAM_FRONT 799장 (1600×900) |
| `hf_cache_processor/` · `preproc_ref.json` | 전처리기 캐시(transformers 4.49.0), 기준 해시(서버에서 만든 것, 이전과 같다) |
| `a100_ref/` | A100 강타입 799장 결과 (꾸러미를 만들 때 있었던 것만) |
| `SHA256SUMS_pkg` · `SHA256SUMS_onnx` | 꾸러미 파일 · ONNX 체크섬 |
