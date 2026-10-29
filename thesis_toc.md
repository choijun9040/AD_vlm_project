<!-- 자동 생성: scripts/build_thesis.py (2026-10-29). 직접 고치지 말 것 — 장별 초안을 고치고 다시 생성한다. -->

# 비전-언어 모델의 저정밀 배포를 위한 표현 범위 기반 오버플로 진단 및 교정

*Representable-Range-Based Overflow Diagnosis and Correction for Low-Precision Deployment of Vision-Language Models*

석사학위논문 목차 · 합본 생성 2026-10-29 · 학교 양식 적용 전

---

# 차 례

국문 초록  
Abstract  
표 차례  
그림 차례


Ⅰ. 서론  
    1. 연구 배경  
    2. 연구 목적 및 기여  
    3. 연구 범위  
    4. 논문의 구성


Ⅱ. 관련 연구  
    1. 주행 VLM의 경량화와 효율화  
    2. 저정밀 배포와 활성 이상치  
    3. 평가 데이터셋  
    4. 선행 연구 공백


Ⅲ. 실험 모델과 평가 방법  
    1. 증류 데이터와 손실 함수  
    2. 비교 변형과 통제 조건  
    3. 평가 프로토콜  
    4. 변형별 정확도 비교 결과


Ⅳ. fp16 표현 범위 초과와 출력 상실  
    1. 표현 범위 여유의 정의와 측정 규칙  
    2. 활성 크기와 해상도별 상실률  
    3. 표준 압축 지표와의 비교  
    4. 오버플로 지점과 곱셈 이전 교정  
    5. 여유의 영향 요인  
    6. 공개 배포 모델의 여유


Ⅴ. TensorRT와 엣지 보드에서의 배포 검증  
    1. 비전 인코더의 ONNX 변환  
    2. 정밀도 조건별 측정 프로토콜  
    3. A100 TensorRT 상실률과 교정 효과  
    4. bf16 전환과 지연 비용  
    5. Jetson Orin Nano 실측 결과


Ⅵ. 배포 전 진단·교정 도구 headroom_guard  
    1. 도구 구성과 설계 판단  
    2. 판정 기준의 데이터 기반 설정  
    3. 도구 검증과 배포 조건 정확도


Ⅶ. 결론  
    1. 요약과 기여  
    2. 한계  
    3. 실무 권고  
    4. 향후 연구  
참고 문헌  
부록  
    A. 측정 환경  
    B. 엣지 보드의 측정 절차  
    C. 공개 저장소 실측 상세

---

# 국문 초록

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
799장)로 예측하였고, 제안 교정은 정확도를 48.76%로 회복하여 bf16 실행과 유의한 차이가 없으면서(11,309문항, McNemar p = 0.24) fp16 형식을
유지하였다. 같은 상실과 교정 효과는 TensorRT 엔진과 Jetson Orin Nano(메모리가 허락한 해상도)에서도 확인되었다.

이상의 결과는 압축된 VLM을 저정밀로 배포하기 전에 배포 해상도에서 표현 범위 여유를 측정하여 출력 상실 위험을
진단하고 교정해야 함을 보여 준다.

**주요어**: 표현 범위 여유, 수치 오버플로, 저정밀 추론, 가중치 교정, 비전-언어 모델, 자율주행


---

# Abstract

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
restores accuracy to 48.76%, not significantly different from bf16 execution (11,309 questions, McNemar p = 0.24), while keeping fp16.
The same collapse and correction were confirmed on TensorRT engines and a Jetson Orin Nano (at the
resolutions its memory allows).

These results show that representable-range headroom must be measured at the deployment
resolution to diagnose and correct the collapse risk before deploying compressed VLMs in low
precision.

**Keywords**: representable-range headroom, numerical overflow, low-precision inference, weight
correction, vision-language model, autonomous driving

\* A thesis for the degree of Master in ○○ 20○○.

---

# 표 차례

표 1. 다섯 학생 변형의 손실·가중치·교사 입력  
표 2. 변형 간 통제 조건  
표 3. 변형별·질문 유형별 NuScenes-QA 검증 정확도  
표 4. 측정 형식에 따른 여유와 판정  
표 5. 형식과 스케일 방식에 따른 위험의 성격  
표 6. 비전 인코더 블록별 최대 활성 크기  
표 7. 입력 해상도별 fp16 상실률  
표 8. 원본 해상도에서의 모델별 여유와 해석  
표 9. 여유·상실률과 표준 압축 지표  
표 10. 양자화 검사 대상과 fp16 실행 대상  
표 11. 마지막 블록 최댓값의 성분 분해  
표 12. 사후 개입 위치에 따른 활성과 상실률  
표 13. 정렬 손실 가중치 λ_align에 따른 활성과 여유  
표 14. 정렬 대상에 따른 마지막 블록 활성  
표 15. 공개 양자화 저장소의 설정 패턴  
표 16. ONNX 내보내기의 제약과 처리  
표 17. ONNX 그래프와 대조 결과  
표 18. TensorRT 정밀도 조건별 상실률·메모리·지연  
표 19. 네 가지 배포 선택지의 비교  
표 20. 실행 경로별 상실률과 초과 비율  
표 21. Orin Nano 실측 결과  
표 22. headroom_guard의 명령  
표 23. 초과 비율과 실제 상실률  
표 24. 도구의 판정 기준  
표 25. 모델 계열별 도구 적용 결과  
표 26. λ 스윕 체크포인트의 여유와 실제 상실  
표 27. 배포 조건 정확도의 종단 검증  
표 28. fp16과 bf16의 짝 정확도 비교  
표 29. 실행 조건에 따른 INT4 정확도  
부록 표 1. 측정 환경  
부록 표 2. transformers 버전에 따른 같은 모델의 측정값 (원본 해상도, 799장, bf16)  
부록 표 3. 위험 구성 공개 저장소의 실측 여유  
부록 표 4. 기반 모델과 양자화 파생본의 여유 상속  
부록 표 5. Qwen 세대·크기별 여유

---

# 그림 차례

그림 1. 문제의 개요  
그림 2. 비전 인코더 블록별 최대 활성의 크기  
그림 3. 입력 해상도에 따른 (a) 마지막 블록 활성 크기와 (b) fp16 상실률  
그림 4. 마지막 블록 fp16 여유와 INT8 양자화 SNR  
그림 5. 정렬 손실 가중치 λ_align에 따른 (a) 블록별 활성 크기와 (b) 마지막 블록 fp16 여유  
그림 6. 실행 경로별 fp16 상실률과 엣지 보드 지연  
그림 7. 최악 이미지 여유에 따른 판정 기준 보정
