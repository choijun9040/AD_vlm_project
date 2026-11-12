# 학위 논문 그림 목록·캡션 (2026-11-06 전면 수정)

> **규칙.** 캡션마다 **표본 · 형식 · 해상도**를 적는다. 그림 속 글자는 영문, 캡션은 국문. 모든 그림은 `scripts/`의 스크립트로
> 결과 파일에서 다시 만들 수 있다. 이전 그림(학생 모델, fig0~fig7)은 원고에서 빠졌고 파일은 `figures/`에 남아 있다.
> 합본 스크립트가 `**그림 N-M.**` 캡션을 읽고 `FIGURES` 목록(scripts/build_thesis.py)의 기준 문장 뒤에 넣는다.

---

## 그림 목록

| 키 | 파일 | 생성 스크립트 | 데이터 |
|---|---|---|---|
| 그림 3-1 | `figures/fig_resid_schematic.png` | `scripts/plot_thesis_v2.py` | (개념도) |
| 그림 5-1 | `figures/fig_sdxl_gen_view.png` | `scripts/cmp_sdxl_view.py` | `eval_results/cmp/sdxl_gen_cmp.latents.pt` (앞 8개) |
| 그림 5-2 | `figures/fig_calib_c.png` | `scripts/plot_thesis_v2.py` | `env4/dlm_gen_qwen.json.calib.json`, `calib100_gemma_*.json`, `dlm_diag_*.json`, `sdxl_calib.json`, `env3/sdxl_gen_cmp.json`, `flux_calib.json`, `flux_cmp.json` |
| 그림 5-3 | `figures/fig_fidelity.png` | `scripts/plot_thesis_v2.py` | `env4/dlm_fid_qwen.json`, `dlm_fid_gemma_*.json`, `env3/sdxl_gen_cmp.json`, `flux_rev35.json` |
| 그림 5-4 | `figures/fig_deploy.png` | `scripts/plot_thesis_v2.py` | `dlm_diag_qwen.json`, `dlm_lat_*.json`, `orin_dlm/results/*.json` |
| 그림 1-1 | `figures/fig_teaser.png` | `scripts/plot_thesis_v2.py` | (a) 형식 규격, (b) `figures/fig_sdxl_gen_view.png` 4번째 열 |
| 그림 3-0 | `figures/fig_flow.png` | `scripts/plot_thesis_v2.py` | (개념도) |
| 그림 3-2 | `figures/fig_region.png` | `scripts/plot_thesis_v2.py` | (모식도) |
| 그림 3-3 | `figures/fig_block_profile.png` | `scripts/plot_thesis_v2.py` | `env6/qwen_block_profile.json` |
| 그림 5-6 | `figures/fig_flux_qual.png` | `scripts/plot_thesis_v2.py` | `env6/flux_qual/` |
| 그림 5-5 | `figures/fig_qwen_qual.png` | `scripts/plot_thesis_v2.py` | `env4/dlm_gen_qwen.json`, `data/nuscenes/samples` |

## 캡션

**그림 3-1.** 잔차 재조정 구조. 잔차 흐름의 기록 층(입력부와 블록의 출력 투영)을 c배, 입력 정규화의 eps를 c²배 하면 정규화 출력이 그대로이므로 fp32 함수는 변경되지 않고 잔차 흐름의 크기만 c배가 된다. 정규화가 아닌 층이 흐름을 입력받는 출력부가 있으면 그 층을 1/c배 한다. c는 진단용 입력의 최댓값에 곱했을 때 32,752(65,504의 절반) 이하가 되는 가장 큰 2의 거듭제곱(1 이하)이다.

**그림 5-1.** SDXL VAE 디코더 복호 결과 예. PartiPrompts 1~8번으로 SDXL base 1.0이 생성한 잠재(1024×1024)를 복호해 256×256으로 축소하여 제시하였다. 위에서부터 원본 VAE fp32, 원본 VAE fp16, SDXL-VAE-FP16-Fix fp16, 잔차 재조정 fp16(c = 1/64)이며, 비유한 출력은 검정으로 표시하였다.

**그림 5-2.** 최댓값과 c. 세로축은 로그 축이고, 점선은 fp16 상한 65,504, 실선은 목표 32,752이다. 진단 집합은 Qwen2.5-VL-3B와 Gemma 4 26B-A4B·31B(비전 토큰 280·1,120)가 DriveLM 학습 100장, SDXL VAE 디코더와 FLUX.1-schnell 변환기가 PartiPrompts 101~200번이다. 막대 위의 값은 c이다.

**그림 5-3.** 원본 fp32 출력 대비 상대 오차 중앙값. 세로축은 로그 축이다. SDXL VAE 디코더는 생성 잠재 100개, FLUX.1-schnell은 PartiPrompts 100개(4스텝, 최종 잠재), Qwen2.5-VL-3B와 Gemma 4는 비전 인코더 출력(DriveLM 검증 799장)이다. 잔차 재조정과 FP16-Fix는 fp16, Draw Things는 공개 설정(잔차 fp32, 구현 코드의 fp16 누산 포함)의 재현, 회색 막대는 bf16이다. 무교정 fp16은 상실하므로 표시하지 않았다. 범례의 RR은 잔차 재조정이다.

**그림 5-4.** 공식 Qwen2.5-VL-3B 비전 인코더 배포 결과. 입력은 DriveLM 검증 799장(896×1596)이다. (a) PyTorch, A100 TensorRT fp16 엔진, Jetson Orin Nano TensorRT fp16 엔진(15W)에서 무교정과 잔차 재조정의 상실. (b) A100 TensorRT 엔진의 지연 p50(다른 작업이 없는 GPU). 범례의 RR은 잔차 재조정이다.

**그림 1-1.** 16비트 부동소수 형식과 fp16 출력 상실. (a) fp32·bf16·fp16의 부호·지수·가수 비트 수와 표현 상한. (b) SDXL VAE 디코더로 같은 생성 잠재(PartiPrompts 4번)를 복호한 결과로, 왼쪽부터 원본 VAE fp32, 원본 VAE fp16(검정은 출력 상실), 잔차 재조정 fp16(c = 1/64)이다. 범례의 RR은 잔차 재조정이다.

**그림 3-0.** 진단과 교정 방법 흐름. 평가와 겹치지 않는 진단 집합을 bf16으로 진단하여 오버플로 지점과 최댓값 M을 찾고, M으로 c를 정한다. 자동 구성 도구는 오버플로 지점에서 출발해 편집할 층을 식별하고 c로 편집을 적용한다. 교정한 모델은 평가 입력에서 fp32 등가성과 fp16 실행의 상실을 확인한다.

**그림 3-2.** 자동 구성 도구 영역 탐색 모식도. 실제 모델이 아닌 예시이다. 목표(32,752)를 초과한 연산(빨강)에서 출발해 c배가 되는 텐서의 영역(파란 띠)을 넓히고, 영역에 값을 만드는 선형층은 기록 층(×c, 입력이 이미 영역 안이면 편향만 ×c), 영역의 값을 입력받는 정규화는 eps ×c², 정규화가 아닌 선형층은 출력부(×1/c)로 식별한다.

**그림 3-3.** Qwen2.5-VL-3B 비전 인코더 블록 출력 최댓값. 진단 집합(DriveLM 학습 100장)을 bf16으로 측정한 블록별 최댓값이며, 세로축은 로그 축이다. 점선은 fp16 상한 65,504, 실선은 목표 32,752이다. 범례의 RR은 잔차 재조정(c = 1/4)이다.

**그림 5-5.** Qwen2.5-VL-3B 생성 답 예. DriveLM 검증 799장의 평가 순서에서 무교정 fp16 비전 인코더가 출력을 상실한 첫 세 문항(0·1·2번)이며, 답의 내용과 무관하게 이 규칙으로 골랐다. 질문은 NuScenes-QA 검증 문항이고, 답은 탐욕 해독으로 최대 128토큰까지 생성한 것을 줄여 표시하였다. 범례의 RR은 잔차 재조정이다.

**그림 5-6.** FLUX.1-schnell 생성 결과 예. PartiPrompts 1~4번(번호 순, 4스텝, 시드 = 번호)을 원본 fp32, fp16 클리핑 끔(검정 = 상실), Draw Things 공개 설정, fp16 + 잔차 재조정(c = 1/8)으로 생성하였다. 이미지 아래 값은 원본 fp32 대비 최종 잠재 상대 오차이다. 범례의 RR은 잔차 재조정이다.

