<!-- 자동 생성: scripts/build_thesis.py (2026-10-24). 직접 고치지 말 것 — 장별 초안을 고치고 다시 생성한다. -->

# 비전-언어 모델의 저정밀 배포를 위한 표현 범위 기반 오버플로 진단 및 교정

*Representable-Range-Based Overflow Diagnosis and Correction for Low-Precision Deployment of Vision-Language Models*

석사학위논문 목차 · 합본 생성 2026-10-24 · 학교 양식 적용 전

---

# 차 례

국문 초록
Abstract
표 차례
그림 차례


Ⅰ. 서론
    1. 연구 배경
    2. 논지와 기여
    3. 연구 범위
    4. 논문의 구성


Ⅱ. 관련 연구
    1. 주행 VLM 경량화와 효율화 연구
    2. 저정밀 배포와 활성 이상치
    3. 평가 프로토콜
    4. 선행 연구에서 비어 있는 축


Ⅲ. 통제된 다중 변형 증류 파이프라인
    1. 증류 파이프라인 구성
    2. 다섯 변형과 공정성 통제
    3. 평가 프로토콜
    4. 변형별 정확도 비교 결과


Ⅳ. fp16 표현 범위 초과와 수치 표현의 상실
    1. 표현 범위 여유의 정의와 측정 규칙
    2. 활성 크기와 이산적 실패
    3. 위험을 예고하지 못하는 표준 압축 지표
    4. 오버플로 지점과 곱셈 이전 교정
    5. 여유를 움직이는 수단


Ⅴ. 이산적 실패와 집계 지표의 무력화
    1. INT4 정확도 1.39%의 원인 규명
    2. 공개 양자화 저장소 조사와 위험 구성
    3. 설정·설계·가중치로는 예측되지 않는 여유


Ⅵ. 저정밀 배포 검증 파이프라인과 실기기 확인
    1. ONNX 내보내기
    2. 정밀도 조건별 측정 프로토콜
    3. TensorRT 측정 결과와 교정 검증
    4. 형식 선택과 비용
    5. Jetson Orin Nano 실측 결과


Ⅶ. 배포 전 진단·교정 도구 headroom_guard
    1. 도구 구성과 설계 판단
    2. 판정 기준의 데이터 기반 설정
    3. 검증과 배포 조건 정확도
    4. 정리


Ⅷ. 종합 논의와 한계
    1. 논지의 검증
    2. 사례 연구의 범위
    3. 한계


Ⅸ. 결론
    1. 요약과 기여
    2. 범위와 남은 문제
    3. 실무 권고
    4. 향후 연구
참고 문헌
부록
    A. 측정 환경
    B. 엣지 보드의 측정 절차
감사의 글

---

# 국문 초록

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

# Abstract

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

---

# 표 차례

표 1. 다섯 학생 변형의 손실·가중치·교사 입력
표 2. 다섯 변형의 공정성 통제 항목
표 3. 변형별·질문 유형별 NuScenes-QA 검증 정확도
표 4. 측정 형식에 따른 여유와 판정
표 5. 비전 인코더 블록별 최대 활성 크기
표 6. 입력 해상도별 fp16 상실률
표 7. 원본 해상도에서의 모델별 여유와 해석
표 8. 여유·상실률과 표준 압축 지표
표 9. 양자화 검사를 받는 부분과 fp16으로 실행되는 부분
표 10. 마지막 블록 최댓값의 성분 분해
표 11. 사후 개입 위치에 따른 활성과 상실률
표 12. 정렬 손실 가중치 λ에 따른 활성과 여유
표 13. 정렬 대상에 따른 마지막 블록 활성
표 14. 교란 요인 제거 전후의 INT4 정확도
표 15. 공개 양자화 저장소의 설정 패턴
표 16. 위험 구성 공개 저장소의 실측 여유
표 17. 기반 모델과 양자화 파생본의 여유 상속
표 18. Qwen 세대별 여유
표 19. ONNX 내보내기의 제약과 처리
표 20. 내보낸 ONNX 그래프와 대조 결과
표 21. TensorRT 정밀도 조건별 상실률·메모리·지연
표 22. 네 가지 배포 선택지의 비교
표 23. 실행 경로별 상실률과 초과 비율
표 24. Orin Nano 실측 결과
표 25. headroom_guard의 명령
표 26. 초과 비율과 실제 상실률
표 27. 도구의 세 판정
표 28. 모델 계열별 도구 적용 결과
표 29. 학습된 λ 체크포인트의 여유와 실제 상실
표 30. 배포 조건 정확도의 종단 검증
표 31. fp16과 bf16의 짝지은 정확도 비교
표 32. 기여별 근거와 강도
표 33. 형식과 스케일 방식에 따른 위험의 성격

---

# 그림 차례

그림 1. 비전 인코더 블록별 최대 활성의 크기
그림 2. 입력 해상도에 따른 (a) 마지막 블록 활성 크기와 (b) fp16 상실률
그림 3. 마지막 블록 fp16 여유와 INT8 양자화 SNR
그림 4. 정렬 손실 가중치 λ_align에 따른 (a) 블록별 활성 크기와 (b) 마지막 블록 fp16 여유
그림 5. 실행 경로별 fp16 상실률과 엣지 보드 지연
그림 6. 최악 이미지 여유에 따른 판정 기준 보정
