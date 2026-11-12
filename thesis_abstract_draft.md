# 학위 논문 초록 초안 (2026-11-06 전면 수정)

> **이 문서의 위치.** 국문·영문 초록. 합본에는 영문 초록만 싣는다(2026-10-30 사용자 결정). 구조는 지도교수 피드백(2026-10-19)의
> 「연구 내용·필요성 → 기존 연구의 문제점 → 제안 → 기존 대비 실험 결과 → 결론」 다섯 단락을 유지한다. 주장은 「원래 출력을 지킨다」이며
> 정확도·화질 향상을 주장하지 않는다(`docs/manuscript_rules.md`). **학교 규정에서 확인할 것**: 분량 상한·단락 수·주요어 개수·영문 초록 형식.
> 이전 초록(학생 모델 중심)은 git 이력에 있다.

---

## 국문 초록

**비전-언어 모델과 확산 모델의 fp16 배포를 위한 오버플로 진단 및 잔차 재조정**

비전-언어 모델(VLM)과 확산 모델을 bf16을 지원하지 않는 엣지 가속기에 배포하려면 fp16을 사용해야 하며, 활성값이 fp16 상한(65,504)을
초과하면 그 입력의 출력 전체가 상실된다. Qwen2.5-VL-3B의 비전 인코더는 DriveLM 검증 799장 중 71.5%, Gemma 4의 비전 인코더는 형식 승격이
없는 배포 경로에서 799장 모두 출력을 상실하였다. SDXL VAE 디코더는 생성 잠재 100개 모두, FLUX.1-schnell은 값 클리핑 없이 프롬프트 100개
모두 출력을 상실하였다.

기존 처방은 연산을 fp32로 승격하거나, 모델을 미세조정하거나(SDXL-VAE-FP16-Fix), 특정 블록을 고정 배수로 축소한다(Draw Things). 이들은
fp32 경로가 필요하거나, 모델의 출력을 변경하거나, 모델마다 조정해야 한다.

본 논문은 평가와 겹치지 않는 입력 100개(진단 집합)를 bf16으로 측정해 오버플로 지점과 최댓값을 찾는 진단과, 그 최댓값으로 정한 배수 c만큼
잔차 흐름을 축소하면서 fp32 함수를 유지하는 학습 없는 **잔차 재조정**을 제안한다. 교정할 층은 연산 그래프에서 자동으로 탐색한다.

잔차 재조정은 모든 대상에서 fp16 출력 상실을 없앴고, 기존 처방보다 원본 fp32 출력에 가까웠다. SDXL VAE에서 원본 출력 대비 PSNR 중앙값은
67.2 dB로 SDXL-VAE-FP16-Fix의 46.1 dB보다 높았고, FLUX.1-schnell에서 최종 잠재 상대 오차 중앙값은 0.0602로 Draw Things 공개 설정의 0.306보다
작았다. VLM에서는 bf16 대비 약 1/3의 오차를 보였고, 교정한 Qwen 비전 인코더는 A100과 Jetson Orin Nano의 TensorRT fp16 엔진에서 모든
이미지의 출력을 유지하였다.

이 결과는 오버플로가 잔차 흐름에 있을 때 학습 없이 원본 출력을 지키면서 fp16 상실을 없앨 수 있음을 보여 준다.

**주요어**: fp16 오버플로, 오버플로 진단, 잔차 재조정, 저정밀 배포, 비전-언어 모델, 확산 모델

---

## Abstract

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
