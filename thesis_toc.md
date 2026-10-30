<!-- 자동 생성: scripts/build_thesis.py (2026-10-30). 직접 고치지 말 것 — 장별 초안을 고치고 다시 생성한다. -->

# 비전-언어 모델의 저정밀 배포를 위한 표현 범위 기반 오버플로 진단 및 교정

*Representable-Range-Based Overflow Diagnosis and Correction for Low-Precision Deployment of Vision-Language Models*

석사학위논문 목차 · 합본 생성 2026-10-30 · 학교 양식 적용 전

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
    1. 주행 VLM의 경량화와 효율화  
    2. 저정밀 배포와 활성 이상치  
    3. 평가 데이터셋


Ⅲ. 진단 및 교정 방법  
    1. 표현 범위 여유  
    2. 곱셈 이전 가중치 교정  
    3. 진단·교정 도구  
    4. 판정 기준  
    5. 배포 검증 절차


Ⅳ. 실험 구성  
    1. 증류 모델과 비교 변형  
    2. 평가 데이터와 측정 규칙  
    3. 실행 환경


Ⅴ. 결과 및 분석  
    1. 변형별 정확도  
    2. 해상도별 출력 상실과 표준 지표  
    3. 진단 지표의 검증  
    4. 오버플로 지점과 교정 효과  
    5. 배포 런타임과 엣지 보드  
    6. 여유의 영향 요인과 공개 배포 모델


Ⅵ. 결론  
참고 문헌  
부록  
    A. 측정 환경  
    B. 엣지 보드의 측정 절차  
    C. 공개 저장소 실측 상세

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
activation magnitude) and proposes a diagnostic that estimates the collapse rate as the fraction of
images exceeding the format limit, together with a training-free correction that scales down the two input projections before the
element-wise multiplication in the gated MLP.

We validated both on 3B students that differ only in their loss terms within a distillation setup
with a Qwen2.5-VL-7B teacher. The students differ in headroom from 0.88× to 2.94× (median over images, native resolution); the student below 1× loses its output on 96.0% of 799 images in fp16.
Its accuracy on 11,309 NuScenes-QA questions at native resolution falls from 48.47% (bf16) to 1.34% (fp16). Per-tensor INT8 quantization SNR rates this student
higher than the collapse-free one (32.28 vs. 22.78 dB). The proposed diagnostic estimates the
collapse rate of six models, including the pretrained model, with a mean absolute error of 1.1 percentage points (PyTorch fp16, native resolution, 799 images), and the proposed correction
restores accuracy to 48.76%, 0.29 percentage points from bf16 execution without overflow (48.47%; 11,309 questions), while keeping fp16.
The same collapse and correction were confirmed on TensorRT engines and, with strongly typed engines
at resolutions up to 802,816 pixels, on a Jetson Orin Nano.

These results show that representable-range headroom must be measured at the deployment
resolution to diagnose and correct the collapse risk before deploying compressed VLMs in low
precision.

**Keywords**: representable-range headroom, numerical overflow, low-precision inference, weight
correction, vision-language model, autonomous driving

\* A thesis for the degree of Master in ○○ 20○○.

---

# 표 차례

표 1. 도구의 판정 기준  
표 2. 다섯 학생 변형의 손실·가중치·교사 입력  
표 3. 측정 형식에 따른 여유와 판정  
표 4. 변형별·질문 유형별 NuScenes-QA 검증 정확도  
표 5. 비전 인코더 블록별 최대 활성 크기  
표 6. 입력 해상도별 fp16 상실률  
표 7. 여유·상실률과 표준 압축 지표  
표 8. 실행 조건에 따른 INT4 정확도  
표 9. 초과 비율과 실제 상실률  
표 10. λ 스윕 체크포인트의 여유와 실제 상실  
표 11. 마지막 블록 최댓값의 성분 분해  
표 12. 사후 개입 위치에 따른 활성과 상실률  
표 13. 모델 계열별 도구 적용 결과  
표 14. 배포 조건 정확도의 종단 검증  
표 15. fp16과 bf16의 짝 정확도 비교  
표 16. 네 가지 배포 선택지의 비교  
표 17. 실행 경로별 상실률과 초과 비율  
표 18. Orin Nano 실측 결과  
표 19. 정렬 손실 가중치 λ_align에 따른 활성과 여유  
표 20. 정렬 대상에 따른 마지막 블록 활성  
표 21. 공개 양자화 저장소의 설정 패턴  
부록 표 1. 측정 환경  
부록 표 2. 위험 구성 공개 저장소의 실측 여유

---

# 그림 차례

그림 1. 문제의 개요  
그림 2. 비전 인코더 블록별 최대 활성의 크기  
그림 3. 입력 해상도에 따른 (a) 마지막 블록 활성 크기와 (b) fp16 상실률  
그림 4. 마지막 블록 fp16 여유와 INT8 양자화 SNR  
그림 5. 최악 이미지 여유에 따른 판정 기준 보정  
그림 6. 실행 경로별 fp16 상실률과 엣지 보드 지연  
그림 7. 정렬 손실 가중치 λ_align에 따른 (a) 블록별 활성 크기와 (b) 마지막 블록 fp16 여유
