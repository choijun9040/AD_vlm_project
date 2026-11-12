<!-- 자동 생성: scripts/build_thesis.py (2026-11-12). 직접 고치지 말 것 — 장별 초안을 고치고 다시 생성한다. -->

# 비전-언어 모델과 확산 모델의 fp16 배포를 위한 오버플로 진단 및 잔차 재조정

*Overflow Diagnosis and Residual Rescaling for fp16 Deployment of Vision-Language and Diffusion Models*

석사학위논문 목차 · 합본 생성 2026-11-12 · 학교 양식 적용 전

---

# 차 례

Abstract  
표 차례  
그림 차례


Ⅰ. 서론  
    1. 연구 배경  
    2. 연구 목적 및 기여  
    3. 논문 구성


Ⅱ. 관련 연구  
    1. 저정밀 형식과 형식 승격  
    2. 활성 이상치와 fp16 오버플로  
    3. 기존 fp16 오버플로 처방  
    4. 평가 데이터셋


Ⅲ. 진단 및 교정 방법  
    1. 출력 상실과 표현 범위 여유  
    2. 잔차 재조정  
    3. 배수 c 결정과 교정 확인  
    4. 자동 구성 도구  
    5. 배포 검증 절차


Ⅳ. 실험 구성  
    1. 대상 모델  
    2. 데이터와 진단 집합  
    3. 지표  
    4. 실행 환경


Ⅴ. 결과 및 분석  
    1. SDXL VAE: SDXL-VAE-FP16-Fix 대비  
    2. FLUX.1-schnell: Draw Things 대비  
    3. 공식 모델 진단과 교정  
    4. 배포 런타임과 엣지 보드  
    5. 자동 구성 도구 결과


Ⅵ. 결론  
참고 문헌  
부록  
    A. 측정 환경  
    B. 엣지 보드 측정 절차

---

# Abstract

**Overflow Diagnosis and Residual Rescaling for fp16 Deployment of
Vision-Language and Diffusion Models**\*

*○○○ (영문 성명, 예: Kim, Ji-Hye)*

*Department of ○○  
Graduate School, Chungbuk National University  
Cheongju, Korea  
Supervised by Professor ○○○*

Deploying vision-language models (VLMs) and diffusion models on edge accelerators without bf16 support requires fp16, and an
activation above the fp16 limit (65,504) destroys the entire output. The vision encoder of Qwen2.5-VL-3B loses its output on 71.5% of
799 DriveLM images, and that of Gemma 4 on all of them along a deployment path without upcasting; the SDXL VAE decoder loses it on all
100 generated latents, and FLUX.1-schnell on all 100 prompts without value clipping.

Existing remedies upcast overflowing operations to fp32, fine-tune the model (SDXL-VAE-FP16-Fix), or rescale fixed blocks (Draw Things);
they need an fp32 path, change the model's output, or must be tuned per model.

This thesis proposes a bf16 diagnostic that locates overflowing operations and their maximum on 100 diagnosis inputs disjoint from the
evaluation data, and **residual rescaling**, a training-free edit that shrinks the residual stream by a factor c set from that maximum
while leaving the fp32 function unchanged. The layers to edit are found automatically from the operation graph.

Residual rescaling removed fp16 output loss on every target and stayed closer to the original fp32 output than existing remedies: a
median PSNR of 67.2 dB versus 46.1 dB for SDXL-VAE-FP16-Fix, and a median final-latent error of 0.0602 versus 0.306 for Draw Things on
FLUX.1-schnell. On the VLMs, its error was about one third of that of bf16, and the corrected Qwen vision encoder kept its output on all
images with TensorRT fp16 engines on an A100 and a Jetson Orin Nano.

When the overflow lies in the residual stream, fp16 output loss can thus be removed without training while preserving the original
output.

**Keywords**: fp16 overflow, overflow diagnosis, residual rescaling, low-precision deployment, vision-language model, diffusion model

\* A thesis for the degree of Master in ○○ 20○○.

---

# 표 차례

표 1. fp16 오버플로 처방 비교  
표 2. 대상 모델에서 잔차 재조정이 변경하는 층  
표 3. 대상 모델과 형식을 변경한 범위  
표 4. 평가 데이터와 진단 집합  
표 5. SDXL VAE 디코더 실행 결과 (생성 잠재 100개, 원본 fp32 출력 대비)  
표 6. COCO val2017 재구성 결과 (5,000장, 원문 조건, LPIPS는 원문 재현 설정)  
표 7. FLUX.1-schnell 변환기 구성별 결과 (PartiPrompts 100개, 4스텝, 원본 fp32 출력 대비)  
표 8. 공식 VLM 진단과 fp16 상실 (진단용 DriveLM 학습 100장, 평가 DriveLM 검증 799장)  
표 9. Qwen2.5-VL-3B 비전 인코더 출력 차이 (799장, 원본 fp32 출력 대비)  
표 10. Gemma 4 비전 인코더 출력 차이 (경로 (ii), 799장, 원본 fp32 출력 대비)  
표 11. A100 TensorRT 엔진 상실과 지연 (공식 Qwen 비전 인코더, 799장, 896×1596)  
표 12. Jetson Orin Nano 상실과 지연 (공식 Qwen 비전 인코더, 799장, 896×1596, 15W)  
표 13. 자동 구성 도구 결과 (다섯 대상, 진단 집합 100개로 구성, 코사인과 상실은 평가 입력)  
부록 표 1. 측정 환경

---

# 그림 차례

그림 1. 16비트 부동소수 형식과 fp16 출력 상실  
그림 2. 진단과 교정 방법 흐름  
그림 3. 잔차 재조정 구조  
그림 4. Qwen2.5-VL-3B 비전 인코더 블록 출력 최댓값  
그림 5. 자동 구성 도구 영역 탐색 모식도  
그림 6. SDXL VAE 디코더 복호 결과 예  
그림 7. FLUX.1-schnell 생성 결과 예  
그림 8. 최댓값과 c  
그림 9. Qwen2.5-VL-3B 생성 답 예  
그림 10. 원본 fp32 출력 대비 상대 오차 중앙값  
그림 11. 공식 Qwen2.5-VL-3B 비전 인코더 배포 결과
