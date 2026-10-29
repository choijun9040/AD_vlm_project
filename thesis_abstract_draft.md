# 학위 논문 초록 초안 (2026-09-29)

> **이 문서의 위치.** 학위 논문 국문·영문 초록 초안이다. 학술대회 요약문(`ksae2026_abstract.md`)은
> **발견**(지표 역전)을 세우는 글이고, 이 초록은 제목대로 **오버플로의 진단과 교정** — 발견, 기전, 공개 모델에서의
> 범위, 도구, 배포 런타임 확인 — 를 요약한다. 근거는 9장 결론 초안의 9.1·9.1절이다.
>
> **작성 규칙.** 수치마다 조건(해상도·통계량)을 짧게 붙인다. 확인된 것은 A100의 TensorRT fp16 엔진까지다.
> Orin은 강타입 엔진으로 확인한 결과 한 문장을 넣었다(6.5절, 2026-10-02). 관행 빌드 불가 등 조건은 본문에 둔다.
> 분량은 **국문 1,415자**(공백 제외)·**영문 545단어**이고 양쪽 다 **8단락·주요어 6개**다(제목·주요어 포함, 2026-10-19 재집계 — 10-02 판의 「1,225자·478단어·7단락」은 지도교수 피드백 재구성과 결과 단락 분리 이전 값이다). **학교 규정에서 확인할 것 넷**: 분량 상한 · 단락 수 제한(한 단락만 받는 규정도 있다) · 주요어 개수(보통 다섯이라 **하나를 줄여야 할 수 있다**) · 영문 초록의 위치와 제목 반복 여부. 규정이 정해지면 맞춘다.
>
> **정확도 문장의 범위.** "다섯 학생이 정확도로 구별되지 않는다"로 쓰지 않는다 — 출력 증류 학생(46.74%)은
> 뚜렷이 낮다(3.4절). 구별되지 않는 것은 특징 증류 세 학생과 기준선(50.30~51.69%)이다.

> **2026-10-19 재구성 (지도교수 피드백).** 초록을 「연구 내용·필요성 → 기존 연구의 문제점 → 제안 → 기존 대비 실험 결과 → 결론」 다섯 단락으로 다시 썼다. 비교 기준은 **기존 평가 지표**(과제 정확도, INT8 양자화 SNR, 이상치 비율)와 **기존 대안**(bf16 형식 전환, 출력 측 축소)이다. 구어 표현(「고치는」·「빼」·「재야만」·「보고서에 드러나지 않는다」)은 학술 표현으로 바꿨다. 같은 날 17:46~18:21에 고친 판(「오버플로 교정」·「수치 오버플로」 주요어, 첫 단락의 오버플로 정의·적용 범위, 결론의 bf16 측정 규칙, 결과 단락 분리 등)은 사용자 결정으로 되돌리고 커밋 339c5e0 판을 쓴다. 그 판 위에 새 제목 기준 검토 넷을 적용했다: 첫 단락에서 현상을 「오버플로」로 부르고, 첫 문장의 범위를 「차량과 같은 엣지 장치」로 넓히고(자율주행은 실험 사례로 남김), 주요어에 「수치 오버플로」·「가중치 교정」을 넣고 「지식 증류」·「배포 검증」을 뺐으며, 결과 단락을 기존 지표의 실패와 제안 방법의 결과 두 단락으로 나눴다. 이어 첫 단락 끝에 자율주행을 사례로 밝히는 문장을 넣었다(적용 범위 주장은 넣지 않음).
>
---

## 국문 초록

**비전-언어 모델의 저정밀 배포를 위한 표현 범위 기반 오버플로 진단 및 교정**

비전-언어 모델(VLM)을 엣지 장치에 배포하려면 지식 증류로 모델을 줄이고 반정밀도(fp16)로 연산하는 것이 일반적이다.
그러나 비전 인코더의 활성값이 fp16 상한(65,504)을 넘으면 그 이미지에 대한 출력 전체가 상실되며, 압축 모델의
평가에 쓰이는 과제 정확도와 양자화 오차 지표는 이 위험을 판별하지 못한다. 양자화 오차 지표는 절대 크기를
보지 않는 스케일 불변량이고, 정확도는 오버플로가 일어나지 않는 조건(bf16, 평가 해상도)에서 측정되기 때문이다.

본 논문은 층별 **표현 범위 여유**(형식 최댓값 ÷ 최대 활성 크기)를 정의하고, 형식 상한을 넘는 이미지의 비율로
출력 상실률을 예측하는 진단 지표와, 게이트형 MLP의 곱셈 이전 단계에서 가중치를 축소하는 학습 없는 교정 방법을
제안한다.

Qwen2.5-VL-7B 교사로부터 손실 함수만 달리하여 증류한 3B 학생 모델로 검증하였다. 학생 모델의 여유는 원본 해상도에서
0.88~2.94배(이미지별 최대의 중앙값)로 달랐고, 여유가 1 미만인 모델은 fp16에서 이미지의
96.0%에 대해 출력을 상실하여 정확도가 48.47%(bf16)에서 1.34%로 떨어졌다. 텐서 단위 INT8 양자화 SNR은 이 모델을
상실이 없는 모델보다 오히려 높게 평가하였다(32.28 대 22.78 dB). 제안 지표는 6개 모델의 상실률을 평균 절대 오차 1.1%p(원본 해상도,
799장)로 예측하였고, 제안 교정은 정확도를 오버플로 없는 bf16 실행(48.47%)과
같은 수준인 48.76%로 회복하면서(11,309문항) fp16 형식을
유지하였다. 같은 상실과 교정 효과는 TensorRT 엔진과 Jetson Orin Nano(메모리가 허락한 해상도)에서도 확인되었다.

이상의 결과는 압축된 VLM을 저정밀로 배포하기 전에 배포 해상도에서 표현 범위 여유를 측정하여 출력 상실 위험을
진단하고 교정해야 함을 보여 준다.

**주요어**: 표현 범위 여유, 수치 오버플로, 저정밀 추론, 가중치 교정, 비전-언어 모델, 자율주행

---

## Abstract

**Representable-Range-Based Overflow Diagnosis and Correction for Low-Precision Deployment of
Vision-Language Models**\*

*○○○ (영문 성명, 예: Kim, Ji-Hye)*

*Department of ○○  
Graduate School, Chungbuk National University  
Cheongju, Korea  
Supervised by Professor ○○○*

Deploying vision-language models (VLMs) on edge devices typically involves both knowledge distillation
and half-precision (fp16) inference. When an activation in the vision encoder exceeds the fp16 limit
(65,504), the model's entire output for that image is lost, yet the task accuracy and
quantization-error metrics used to evaluate compressed models cannot detect this risk:
quantization-error metrics are scale-invariant and ignore absolute magnitude, and accuracy is
measured under conditions where overflow does not occur (bf16, evaluation resolution).

This thesis defines a per-layer **representable-range headroom** (format maximum ÷ maximum
activation magnitude) and proposes a diagnostic that predicts the collapse rate as the fraction of
images exceeding the format limit, together with a training-free correction that scales the
weights before the multiplication in the gated MLP.

We validated both on 3B students distilled from a Qwen2.5-VL-7B teacher that differ only in their loss
functions. The students differ in headroom from 0.88× to 2.94× (median over images, native resolution); the student below 1× loses its output on 96.0% of images in fp16, and
its accuracy falls from 48.47% (bf16) to 1.34%. Per-tensor INT8 quantization SNR rates this student
higher than the collapse-free one (32.28 vs. 22.78 dB). The proposed diagnostic predicts the
collapse rate of six models with a mean absolute error of 1.1 percentage points (native resolution, 799 images), and the proposed correction
restores accuracy to 48.76%, on par with bf16 execution without overflow (48.47%; 11,309 questions), while keeping fp16.
The same collapse and correction were confirmed on TensorRT engines and a Jetson Orin Nano (at the
resolutions its memory allows).

These results show that representable-range headroom must be measured at the deployment
resolution to diagnose and correct the collapse risk before deploying compressed VLMs in low
precision.

**Keywords**: representable-range headroom, numerical overflow, low-precision inference, weight
correction, vision-language model, autonomous driving

\* A thesis for the degree of Master in ○○ 20○○.
