<!-- 부록 초안 (2026-10-22 신설, 2026-11-06 전면 수정 #11). 재현에 필요한 측정 환경과 보드 측정 절차. 옛 부록 C(공개 양자화 저장소 실측)는
     원고 범위 밖이 되어 뺐다(git 이력). scripts/build_thesis.py가 참고 문헌 뒤에 넣는다.
     소제목은 `##`로 두면 본문 절로 세어진다 — 합본 스크립트가 부록 절도 차례에 넣으므로 그대로 둔다. -->

---

# 부록

## A. 측정 환경

측정에 사용한 소프트웨어와 하드웨어는 <부록 표 1>과 같다.

**<부록 표 1>** 측정 환경

| 용도 | 환경 |
|---|---|
| Qwen2.5-VL-3B PyTorch 측정 | Python 3.10, PyTorch 2.3.0+cu121, `transformers` 4.49.0, NVIDIA A100-SXM4-80GB |
| Gemma 4 PyTorch 측정 | PyTorch 2.11.0+cu128, `transformers` 5.20.2, NVIDIA A100-SXM4-80GB |
| SDXL VAE·FLUX.1-schnell 측정 | Python 3.10, PyTorch 2.11.0+cu128, `diffusers` 0.41.0, `transformers` 5.20.2, `lpips` 0.1.4, `scikit-image` 0.25.2, NVIDIA A100-SXM4-80GB |
| TensorRT 측정 (A100) | TensorRT 10.13.3.9 |
| 엣지 보드 | Jetson Orin Nano Super 개발자 키트 8GB, L4T R36.4.7, TensorRT 10.3.0.30, 15W 모드 |

공식 Qwen2.5-VL-3B는 Hugging Face 저장소의 main 판(`66285546`)을 사용하였고, Gemma 4 비전 인코더 가중치는 공식 판과 해시가 일치함을 확인하였다.
원본 SDXL VAE는 `stabilityai/sdxl-vae`의 판 `6f5909a7`, SDXL-VAE-FP16-Fix는 `madebyollin/sdxl-vae-fp16-fix`의 판 `207b116d`,
FLUX.1-schnell은 `black-forest-labs/FLUX.1-schnell`의 판 `741f7c3c`를 사용하였다.

## B. 엣지 보드 측정 절차

입력으로는 A100에서 공식 처리기로 전처리한 텐서(799장, 896×1596)를 ONNX 그래프와 함께 보드로 옮겨, SHA-256 체크섬으로 동일한 파일임을
확인한 뒤 사용하였다. 엔진은 보드에서 TensorRT fp16 엔진으로 빌드하였다. 비전 인코더는 블록 16을 경계로 두 엔진으로 분할하여 빌드하고, 측정
시 두 분할 엔진을 별도 프로세스에서 차례로 실행하였다. 지연은 두 분할 엔진의 합이고, 메모리 정점은 측정 동안 보드 전체 메모리 사용량의
최댓값이다.
