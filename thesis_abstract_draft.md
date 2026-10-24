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

비전-언어 모델(VLM)을 차량과 같은 엣지 장치에 배포하려면 지식 증류를 통한 모델 경량화와
반정밀도(fp16) 연산이 함께 요구된다. fp16은 표현할 수 있는 최댓값이 65,504로 제한되어, 비전
인코더의 활성값이 이를 초과하는 오버플로가 일어나면 해당 이미지에 대한 모델 출력 전체가
상실된다. 따라서 압축 모델을 저정밀로 배포하기 전에 이러한 출력 상실 위험을 판별하는 방법이
필요하다. 본 논문은 자율주행 VLM을 사례로 이 문제를 다룬다.

기존 연구는 압축 모델의 배포 적합성을 과제 정확도와 양자화 오차 지표(양자화 신호대잡음비(SNR), 이상치
비율)로 평가한다. 그러나 이 지표들은 **활성값의 절대 크기가 fp16 상한을 넘어 출력이
상실되는지**를 판별하지 못한다. 양자화 오차 지표는 텐서 내 값의 상대적 분포만을 반영하는 스케일
불변량이어서 절대 크기를 보지 않고, 과제 정확도는 평가 해상도와 bf16 형식, 즉 오버플로가
일어나지 않는 조건에서 측정되기 때문이다. 또한 실무에서는 언어 모델을 INT4로 양자화하면서 비전
인코더는 양자화 대상에서 제외하여 fp16으로 실행하는 구성이 널리 쓰이므로, 양자화 오차 평가는
정작 fp16으로 실행되는 비전 인코더에 적용되지 않는다.

본 논문은 이를 해결하기 위해 층별 **표현 범위 여유**(형식 최댓값 ÷ 최대 활성 크기)를 정의하고,
배포 형식의 상한을 초과하는 이미지의 비율로 출력 상실률을 예측하는 진단 지표를 제안한다. 아울러
오버플로 발생 지점의 분석에 근거하여, 게이트형 MLP의 곱셈 이전 단계에서 가중치를 축소하는 학습
없는 교정 방법을 제안하고, 이를 배포 전 진단·교정 도구로 구현하였다.

Qwen2.5-VL-7B 교사 모델로부터 손실 함수만을 달리한 다섯 개의 3B 학생 모델을 증류하여 검증하였다.
특징 증류 모델과 증류 없는 기준 모델의 정확도는 50.30~51.69%로 구별되지 않았으나, 원본
해상도에서 비전 인코더 마지막 블록의 여유는 0.88~2.94배로 달랐으며, 여유가 1 미만인 모델은
fp16에서 이미지의 96.0%에 대해 출력이 상실되어 정확도가 48.47%(bf16)에서 1.34%로 하락하였다.
기존 지표인 텐서 단위 INT8 양자화 SNR은 상실률 96.0%인 모델을 32.28 dB, 0%인 모델을 22.78 dB로
평가하여 출력 상실 위험을 예고하지 못하였고, 이상치 비율도
상실률과 일관된 관계를 보이지 않았다.

반면 제안 지표는 6개 모델의 상실률을 평균 1.1%p의 오차로 예측하였다. 제안 교정 방법은 출력
상실을 제거하여 배포 조건의 정확도를 1.34%에서 48.76%로 회복하였으며, 이는 오버플로가 없는 bf16
실행(48.47%)과 통계적으로 유의한 차이가 없었다. 기존 대안인 bf16 형식 전환은 A100의 TensorRT에서 fp16 대비
1.30~1.36배의 추론 지연을 유발하나, 제안 방법은 fp16 형식을 유지하여 이러한 지연 증가를
피하였다. 같은 출력 상실과 교정 효과는 TensorRT fp16 엔진과 Jetson Orin Nano 8GB의 강타입
엔진(메모리가 허락한 해상도)에서도 확인되었다. 또한 특징 정렬 손실의 가중치를 높이면 여유가 1.79배에서 4.61배로
증가하여(평가 해상도), 증류 설계 단계에서도 여유를 확보할 수 있음을 보였다.

이상의 결과는 압축된 VLM의 저정밀 배포 적합성을 정확도와 양자화 오차 지표만으로 판단할 수
없으며, 배포 해상도에서 표현 범위 여유를 측정하여(크기는 bf16, 출력 상실은 배포 형식에서) 출력 상실 위험을 사전에 진단하고
교정해야 함을 보여 준다.

**주요어**: 표현 범위 여유, 수치 오버플로, 저정밀 추론, 가중치 교정, 비전-언어 모델, 자율주행

---

## Abstract

**Representable-Range-Based Overflow Diagnosis and Correction for Low-Precision Deployment of
Vision-Language Models**

Deploying vision-language models (VLMs) on edge devices such as vehicles requires both model
compression through knowledge distillation and half-precision (fp16) inference. Because fp16 can
represent magnitudes only up to 65,504, an activation in the vision encoder that exceeds this
limit (an overflow) destroys the model's entire output for that image. A method to identify this
risk before low-precision deployment is therefore needed. This thesis addresses the problem
using autonomous-driving VLMs as a case study.

Existing work judges deployment readiness by task accuracy and quantization-error metrics such
as quantization signal-to-noise ratio (SNR) and outlier ratio. These metrics, however, cannot
tell **whether activation magnitudes exceed the fp16 limit and destroy the output**.
Quantization-error metrics are scale-invariant, reflecting only the relative distribution of
values within a tensor, so they ignore absolute magnitude; task accuracy is measured at
evaluation resolution in bf16, where overflow does not occur. Moreover, common practice
quantizes the language model to 4 bits while excluding the vision encoder from quantization and
running it in fp16, so quantization-error evaluation is never applied to the component that
actually runs in fp16.

To address this, we define a per-layer **representable-range headroom** (format maximum ÷
maximum activation magnitude) and propose a diagnostic that predicts the collapse rate as the
fraction of images exceeding the target format's limit. Based on an analysis of where the
overflow occurs, we further propose a training-free correction that scales the weights before
the multiplication in the gated MLP, and implement both as a pre-deployment diagnosis and
correction tool.

We validated the approach on five 3B students distilled from a Qwen2.5-VL-7B teacher that differ
only in their loss functions. The feature-distilled students and the non-distilled baseline are
indistinguishable in accuracy (50.30–51.69%), yet the headroom of the vision encoder's last
block at native resolution ranges from 0.88× to 2.94×; the student below 1× collapses on 96.0%
of images in fp16, and its accuracy falls from 48.47% (bf16) to 1.34%. The existing per-tensor
INT8 quantization SNR rates the student with a 96.0% collapse rate at 32.28 dB and the one with
0% at 22.78 dB, failing to anticipate the risk, and the
outlier ratio shows no consistent relation to collapse either.

The proposed diagnostic predicts the collapse rate of six models with a mean error of 1.1
percentage points. The proposed correction eliminates the collapse and restores
deployment-condition accuracy from 1.34% to 48.76%, with no statistically significant difference
from overflow-free bf16 execution (48.47%). Whereas the existing alternative of switching to
bf16 increases latency by 1.30–1.36× relative to fp16 in TensorRT on an A100, the proposed method keeps fp16 and
avoids this cost. The same collapse and correction were confirmed on a TensorRT fp16 engine and
on a strongly typed engine on a Jetson Orin Nano 8GB (at the resolutions its memory allows). Raising the weight of the
feature-alignment loss also increases headroom from 1.79× to 4.61× (evaluation resolution),
showing that headroom can also be secured at the distillation design stage.

These results show that low-precision deployment readiness of compressed VLMs cannot be judged
by accuracy and quantization-error metrics alone, and that representable-range headroom must be
measured at the deployment resolution (magnitude in bf16, collapse in the deployment format) to diagnose and correct the collapse risk
beforehand.

**Keywords**: representable-range headroom, numerical overflow, low-precision inference, weight
correction, vision-language model, autonomous driving 