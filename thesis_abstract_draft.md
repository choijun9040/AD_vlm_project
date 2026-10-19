# 학위 논문 초록 초안 (2026-09-29)

> **이 문서의 위치.** 학위 논문 국문·영문 초록 초안이다. 학술대회 요약문(`ksae2026_abstract.md`)은
> **발견**(지표 역전)을 세우는 글이고, 이 초록은 제목대로 **측정과 진단 체계** — 발견, 기전, 공개 모델에서의
> 범위, 도구, 배포 런타임 확인 — 를 요약한다. 근거는 9장 결론 초안의 9.1·9.1절이다.
>
> **작성 규칙.** 수치마다 조건(해상도·통계량)을 짧게 붙인다. 확인된 것은 A100의 TensorRT fp16 엔진까지다.
> Orin은 강타입 엔진으로 확인한 결과 한 문장을 넣었다(6.5절, 2026-10-02). 관행 빌드 불가 등 조건은 본문에 둔다.
> 분량은 **국문 1,640자**(공백 제외)·**영문 647단어**이고 양쪽 다 **8단락·주요어 6개**다(제목·주요어 포함, 2026-10-19 재집계 — 10-02 판의 「1,225자·478단어·7단락」은 지도교수 피드백 재구성 이전 값이다). **학교 규정에서 확인할 것 넷**: 분량 상한 · 단락 수 제한(한 단락만 받는 규정도 있다) · 주요어 개수(보통 다섯이라 **하나를 줄여야 할 수 있다**) · 영문 초록의 위치와 제목 반복 여부. 규정이 정해지면 맞춘다.
>
> **정확도 문장의 범위.** "다섯 학생이 정확도로 구별되지 않는다"로 쓰지 않는다 — 출력 증류 학생(46.74%)은
> 뚜렷이 낮다(3.4절). 구별되지 않는 것은 특징 증류 세 학생과 기준선(50.30~51.69%)이다.

> **2026-10-19 재구성 (지도교수 피드백).** 초록을 「연구 내용·필요성 → 기존 연구의 문제점 → 제안 → 기존 대비 실험 결과 → 결론」 다섯 단락으로 다시 썼다. 비교 기준은 **기존 평가 지표**(과제 정확도, INT8 양자화 SNR, 이상치 비율)와 **기존 대안**(bf16 형식 전환, 출력 측 축소)이다. 구어 표현(「고치는」·「빼」·「재야만」·「보고서에 드러나지 않는다」)은 학술 표현으로 바꿨다. 같은 날 새 제목에 맞춰 첫 단락에서 「fp16 오버플로」·「output collapse」를 정의하고, 필요성을 「진단하고 교정하는」 방법으로, 적용 범위(방법은 VLM 일반·위험 실측은 Qwen2.5-VL 세대)를 첫 단락으로 옮겼다. 이어진 검토에서 결론의 「배포 형식에서 여유를 측정」을 bf16 측정 규칙에 맞게 고치고, 「6개 모델」의 구성·여유의 통계량·정확도의 해상도·Orin의 해상도·출력 측 축소와의 대조를 넣고, 「널리 쓰이므로」를 근거(13.7%, 하한)에 맞게 낮추고, 위험 실측 범위 문장을 결과 단락 끝으로 옮겼다. 새 제목 기준 검토로 첫 문장의 범위를 「차량과 같은 엣지 장치」로 넓히고(자율주행은 사례 문장에 남김), 주요어를 제목의 두 축에 맞춰 「수치 오버플로」·「가중치 교정」을 넣고 「지식 증류」·「오버플로 교정」을 뺐으며, 결과 단락을 기존 지표의 실패(앞)와 제안 방법의 결과(뒤) 두 단락으로 나눴다.
>
---

## 국문 초록

**비전-언어 모델의 저정밀 배포를 위한 표현 범위 기반 오버플로 진단 및 교정**

비전-언어 모델(VLM)을 차량과 같은 엣지 장치에 배포하려면 지식 증류를 통한 모델 경량화와
반정밀도(fp16) 연산이 함께 요구된다. fp16은 표현할 수 있는 최댓값이 65,504로 제한되어, 비전
인코더의 활성값이 이를 초과하면(**fp16 오버플로**) 해당 이미지에 대한 모델 출력 전체가 상실된다.
따라서 압축 모델을 저정밀로 배포하기 전에 이러한 출력 상실 위험을 진단하고 교정하는 방법이
필요하다. 자율주행 VLM은 이 문제의 사례이며, 제안하는 진단과 교정은 비전 인코더를 저정밀로
실행하는 VLM에 계열과 무관하게 적용된다.

기존 연구는 압축 모델의 배포 적합성을 과제 정확도와 양자화 오차 지표(양자화 신호대잡음비, 이상치
비율)로 평가한다. 그러나 이 지표들은 **활성값의 절대 크기가 fp16 상한을 넘어 출력이
상실되는지**를 판별하지 못한다. 양자화 오차 지표는 텐서 내 값의 상대적 분포만을 반영하는 스케일
불변량이어서 절대 크기를 보지 않고, 과제 정확도는 평가 해상도와 bf16 형식, 즉 오버플로가
일어나지 않는 조건에서 측정되기 때문이다. 또한 실무에서는 언어 모델을 4비트로 양자화하면서 비전
인코더는 양자화 대상에서 제외하여 fp16으로 실행하는 구성이 쓰이는데, 이 경우 양자화 오차 평가는
정작 fp16으로 실행되는 비전 인코더에 적용되지 않는다.

본 논문은 이를 해결하기 위해 층별 **표현 범위 여유**(형식 최댓값 ÷ 최대 활성 크기)를 정의하고,
배포 형식의 상한을 초과하는 이미지의 비율로 출력 상실률을 예측하는 진단 지표를 제안한다. 아울러
오버플로 발생 지점의 분석에 근거하여, 게이트형 MLP의 곱셈 이전 단계에서 가중치를 축소하는 학습
없는 교정 방법을 제안하고, 이를 배포 전 진단·교정 도구로 구현하였다. 이 교정은 해당 연산을
대체하는 기존 수치 안정화 기법과 달리 가중치 편집만으로 이루어져 추론 그래프와 런타임을 바꾸지
않는다.

Qwen2.5-VL-7B 교사 모델로부터 손실 함수만을 달리한 다섯 개의 3B 학생 모델을 증류하여 검증하였다.
특징 증류 모델과 증류 없는 기준 모델의 정확도는 50.30~51.69%로 구별되지 않았으나, 원본
해상도에서 비전 인코더 마지막 블록의 여유(이미지 중앙값)는 0.88~2.92배로 달랐으며, 여유가 1
미만인 모델은 fp16에서 이미지의 94.8%에 대해 출력이 상실되어 원본 해상도의 정확도가
49.95%(bf16)에서 1.36%로 하락하였다. 기존 지표인 텐서 단위 INT8 양자화 SNR은 상실률 94.8%인
모델을 32.23 dB, 0%인 모델을 22.88 dB로 평가하여 위험 순위를 정반대로 제시하였고(사전학습 모델과
다섯 학생의 6개 모델, 스피어만 순위상관 −1.000), 이상치 비율은 상실률과 일관된 관계를 보이지
않았다.

반면 제안 지표는 6개 모델의 상실률을 평균 2.1%p의 오차로 예측하였다. 제안 교정 방법은 출력 측
축소로는 제거되지 않던(94.8%) 출력 상실을 제거하여 배포 조건의 정확도를 1.36%에서 50.73%로
회복하였으며, 이는 오버플로가 없는 bf16 실행(49.95%)과 통계적으로 유의한 차이가 없었다. 기존
대안인 bf16 형식 전환은 fp16 대비 1.34~1.38배의 추론 지연을 유발하나, 제안 방법은 fp16 형식을
유지하여 이러한 지연 증가를 피하였다. 같은 출력 상실과 교정 효과는 TensorRT fp16 엔진과 Jetson
Orin Nano 8GB의 강타입 엔진(축소 해상도)에서도 확인되었다. 또한 특징 정렬 손실의 가중치를 높이면
여유가 1.79배에서 4.61배로 증가하여(평가 해상도), 증류 설계 단계에서도 여유를 확보할 수 있음을
보였다. 다만 여러 계열의 공개 모델을 포함해 측정한 모델 가운데 상한에 근접한 활성은 Qwen2.5-VL
세대에서만 관찰되었다.

이상의 결과는 압축된 VLM의 저정밀 배포 적합성을 정확도와 양자화 오차 지표만으로 판단할 수
없으며, 배포 해상도에서 넘치지 않는 형식으로 활성 크기를 측정하여 배포 형식에 대한 표현 범위
여유를 구하고, 이로써 출력 상실 위험을 사전에 진단하고 교정해야 함을 보여준다.

**주요어**: 표현 범위 여유, 수치 오버플로, 저정밀 추론, 가중치 교정, 비전-언어 모델, 자율주행

---

## Abstract

**Representable-Range-Based Overflow Diagnosis and Correction for Low-Precision Deployment of
Vision-Language Models**

Deploying vision-language models (VLMs) on edge devices such as vehicles requires both model
compression through knowledge distillation and half-precision (fp16) inference. Because fp16 can
represent magnitudes only up to 65,504, an activation in the vision encoder that exceeds this
limit (**fp16 overflow**) destroys the model's entire output for that image (output collapse). A
method to diagnose and correct this risk before low-precision deployment is therefore needed.
Autonomous-driving VLMs serve as the case study; the proposed diagnosis and correction apply to
any VLM that runs its vision encoder at low precision, regardless of model family.

Existing work judges deployment readiness by task accuracy and quantization-error metrics such
as quantization signal-to-noise ratio (SNR) and outlier ratio. These metrics, however, cannot
tell **whether activation magnitudes exceed the fp16 limit and destroy the output**.
Quantization-error metrics are scale-invariant, reflecting only the relative distribution of
values within a tensor, so they ignore absolute magnitude; task accuracy is measured at
evaluation resolution in bf16, where overflow does not occur. Moreover, a configuration used in
practice quantizes the language model to 4 bits while excluding the vision encoder from
quantization and running it in fp16; in that case quantization-error evaluation is never applied
to the component that actually runs in fp16.

To address this, we define a per-layer **representable-range headroom** (format maximum ÷
maximum activation magnitude) and propose a diagnostic that predicts the collapse rate as the
fraction of images exceeding the target format's limit. Based on an analysis of where the
overflow occurs, we further propose a training-free correction that scales the weights before
the multiplication in the gated MLP, and implement both as a pre-deployment diagnosis and
correction tool. Unlike existing numerical-stability techniques that replace the offending
operation, the correction is a weight edit alone and leaves the inference graph and runtime
unchanged.

We validated the approach on five 3B students distilled from a Qwen2.5-VL-7B teacher that differ
only in their loss functions. The feature-distilled students and the non-distilled baseline are
indistinguishable in accuracy (50.30–51.69%), yet the median per-image headroom of the vision
encoder's last block at native resolution ranges from 0.88× to 2.92×; the student below 1×
collapses on 94.8% of images in fp16, and its native-resolution accuracy falls from 49.95%
(bf16) to 1.36%. The existing per-tensor INT8 quantization SNR rates the student with a 94.8%
collapse rate at 32.23 dB and the one with 0% at 22.88 dB, ranking the risk in exact reverse
(six models: the pretrained model and the five students; Spearman −1.000), and the outlier
ratio shows no consistent relation to collapse.

The proposed diagnostic predicts the collapse rate of six models with a mean error of 2.1
percentage points. The proposed correction eliminates the collapse, which scaling the MLP output
instead leaves at 94.8%, and restores deployment-condition accuracy from 1.36% to 50.73%, with
no statistically significant difference from overflow-free bf16 execution (49.95%). Whereas the
existing alternative of switching to bf16 increases latency by 1.34–1.38× relative to fp16, the
proposed method keeps fp16 and avoids this cost. The same collapse and correction were confirmed
on a TensorRT fp16 engine and on a strongly typed engine on a Jetson Orin Nano 8GB (at reduced
resolution). Raising the weight of the feature-alignment loss also increases headroom from 1.79×
to 4.61× (evaluation resolution), showing that headroom can also be secured at the distillation
design stage. Among the models measured, including public models from several families,
activations near the limit appeared only in the Qwen2.5-VL generation.

These results show that low-precision deployment readiness of compressed VLMs cannot be judged
by accuracy and quantization-error metrics alone, and that the collapse risk must be diagnosed
and corrected beforehand by computing representable-range headroom against the deployment
format's limit, with activation magnitudes measured at the deployment resolution in a
non-overflowing format.

**Keywords**: representable-range headroom, numerical overflow, low-precision inference, weight
correction, vision-language model, autonomous driving 