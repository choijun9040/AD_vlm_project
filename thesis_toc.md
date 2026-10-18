<!-- 자동 생성: scripts/build_thesis.py (2026-10-18). 직접 고치지 말 것 — 장별 초안을 고치고 다시 생성한다. -->

# 자율주행 비전-언어 모델의 저정밀 배포를 위한 표현 범위 여유 측정과 진단

*Measuring and Diagnosing Representable-Range Headroom for Low-Precision Deployment of Autonomous-Driving Vision-Language Models*

석사학위논문 목차 · 합본 생성 2026-10-18 · 학교 양식 적용 전

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
    3. 표준 압축 지표와의 역전
    4. 오버플로 지점과 증폭 구조
    5. 여유를 움직이는 수단


Ⅴ. 이산적 실패와 집계 지표의 무력화
    1. INT4 정확도 1.39%의 원인 규명
    2. 공개 양자화 저장소 조사와 위험 구성
    3. 여유 예측 시도와 한계
    4. 측정 자체의 난점


Ⅵ. 저정밀 배포 검증 파이프라인과 실기기 확인
    1. ONNX 내보내기
    2. 정밀도 조건별 측정 프로토콜
    3. TensorRT 측정 결과와 처방 검증
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
감사의 글

---

# 국문 초록

**자율주행 비전-언어 모델의 저정밀 배포를 위한 표현 범위 여유 측정과 진단**

자율주행용 비전-언어 모델(VLM)을 차량에 탑재하려면 지식 증류로 모델을 줄이고 반정밀도(fp16)로 연산을
줄여야 한다. 두 단계는 각자의 지표 — 증류는 과제 정확도, 양자화는 이상치 비율과 양자화 신호대잡음비(SNR)
— 로 평가된다. 본 논문은 이 평가 체계가 놓치는 실패로 **fp16 표현 범위 초과**를 제시하고, 그것을 배포 전에
재고 고치는 절차를 제안한다.

Qwen2.5-VL-7B 교사를 3B 학생으로 증류하면서 데이터·LoRA 구성·학습 스텝을 고정하고 손실 항만 달리한 다섯
학생을 만들었다. 특징 증류 학생들과 증류 없는 기준선은 정확도로 서로 구별되지 않았으나(50.30~51.69%),
비전 인코더 마지막 블록의 **표현 범위 여유**
(format_max ÷ max|activation|)는 원본 해상도에서 0.88배부터 2.92배까지 갈렸다(이미지별 최대의 중앙값). 여유가
1배 아래인 학생은 fp16에서 이미지의 94.8%에 대해 출력을 통째로 잃었고, 과제 정확도가 49.95%(bf16)에서
1.36%(fp16)로 떨어졌다. 같은 두 학생의 INT8 양자화 SNR은 출력을 잃는 쪽이 32.23 dB, 출력을 잃지 않는 쪽이
22.88 dB로, **표준 압축 지표는 위험을 역방향으로 가리켰다.**

이는 양자화 도구가 검사하는 부분과 양자화에서 제외되어 fp16으로 실행되는 부분이 서로 다르기 때문이다. 실무에서는 언어 모델 백본을 4비트로 강하게
양자화하면서도 **비전 인코더는 양자화에서 빼 fp16/bf16으로 두는** 구성이 흔하다. 양자화 도구는 자기가
누른 층만 보고하므로, 비전 인코더는 보고서에 "안전"으로 적히는 것이 아니라 **항목 자체가 없다.** 따라서
이 위험은 어떤 지표에도 남지 않으며, **배포할 형식과 해상도에서 그 층을 직접 재야만 드러난다.**

오버플로 지점은 마지막 블록 MLP의 출력 투영 안, 사전학습 가중치에 고정된 한 채널로 특정되었다. 곱이
만들어지기 전에 입력을 줄이면 학습 없이 상실이 사라졌고, 출력을 줄이면 사라지지 않았다. 여유는 특징 정렬
손실의 가중치에 따라 단조롭게 변했고, 정렬 대상을 학생 자신으로 바꿔도 효과의 절반이 남았으며, 저장소
설정·인코더 설계·가중치·모델 계열 어느 것으로도 예측되지 않았다. 비전 인코더를 양자화에서 뺀 공개 파생본은
기반 모델의 여유를 그대로 물려받았다.

이 진단을 도구 `headroom_guard`로 구현했다. 도구는 활성 크기를 넘치지 않는 형식(bf16)에서 재고, 형식
상한을 넘는 이미지의 비율로 상실률을 추정하며(PyTorch fp16 기준 6개 모델 평균 오차 2.1%p), 대상이
지원하는 형식에 맞춰 형식 전환이나 가중치 교정을 권고한다. 배포 조건에서 도구가 교정한 가중치는 정확도를 1.36%에서 50.73%로 되돌렸다. 비전
인코더를 ONNX로 내보내 TensorRT fp16 엔진으로 실행해도 같은 상실이 재현되었다. 엣지 보드(Jetson Orin Nano
8GB)의 강타입 엔진도 A100과 같은 비율로 출력을 잃었고(802,816 해상도, 24.4%), 교정 엔진은 출력을 잃지 않았다.

압축된 주행 VLM의 저정밀 배포에서 부족한 것은 여유를 움직이는 수단이 아니라 **그것을 재는 일**이다.

**주요어**: 표현 범위 여유, 저정밀 추론, 지식 증류, 비전-언어 모델, 자율주행, 배포 검증


---

# Abstract

**Measuring and Diagnosing Representable-Range Headroom for Low-Precision Deployment of
Autonomous-Driving Vision-Language Models**

Deploying vision-language models (VLMs) in vehicles requires both knowledge distillation and half-precision
(fp16) inference, and each step is judged by its own metric: task accuracy for distillation, outlier ratio
and quantization signal-to-noise ratio (SNR) for quantization. This thesis identifies a failure that these
metrics miss — **fp16 representable-range overflow** — and proposes a procedure to measure and correct it
before deployment.

We distilled a Qwen2.5-VL-7B teacher into five 3B students that differ only in their loss terms, with data,
LoRA configuration and training steps held fixed. The feature-distilled students and the non-distilled
baseline are indistinguishable in accuracy (50.30–51.69%), yet the
**representable-range headroom** (format_max ÷ max|activation|) of the vision encoder's last block ranges from
0.88× to 2.92× at native resolution (median over images). The student below 1× loses its output entirely on
94.8% of images in fp16, and its task accuracy falls from 49.95% (bf16) to 1.36% (fp16). For the same two
students, INT8 quantization SNR is 32.23 dB for the one that collapses and 22.88 dB for the one that does not:
**the standard compression metric ranks the risk in reverse.**

This happens because the part that quantization tools inspect differs from the part that is excluded from quantization and runs in fp16. In practice it is
common to quantize the language backbone aggressively to 4 bits while **leaving the vision encoder
unquantized in fp16/bf16**. Because a quantization tool reports only the layers it touched, the vision
encoder is not listed as "safe" — **it is absent from the report entirely.** The risk therefore appears in
no metric: **it surfaces only by measuring that layer directly, in the format and at the resolution of
deployment.**

The overflow is localized to a single channel of the last block's MLP output projection, fixed in the
pretrained weights. Scaling the inputs before their product removes the collapse without training; scaling
the output does not. Headroom varies monotonically with the feature-alignment loss weight, half of the effect
remains when the student aligns to itself instead of the teacher, and no repository setting, encoder design,
weight statistic, or model family predicts it. Public derivatives that exclude the vision encoder from
quantization inherit their base model's headroom exactly.

We implement the diagnosis as a tool, `headroom_guard`. It measures magnitudes in a format that does not
overflow (bf16), estimates the collapse rate as the fraction of images exceeding the format limit (mean error
2.1 percentage points over six models in PyTorch fp16), and recommends a format switch or weight correction according to the
formats the target supports. Under deployment conditions, the tool's corrected weights restore accuracy from
1.36% to 50.73%. The same collapse reproduces when the vision encoder is exported to ONNX and run as a TensorRT
fp16 engine. On an edge board (Jetson Orin Nano 8GB), a strongly typed engine collapsed at the same rate as on the
A100 (24.4% at 802,816 pixels), and the corrected engine did not collapse.

What low-precision deployment of compressed driving VLMs lacks is not the means to move headroom, but
**measuring it**.

**Keywords**: representable-range headroom, low-precision inference, knowledge distillation,
vision-language model, autonomous driving, deployment verification

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
표 19. ONNX 내보내기에서 막힌 곳과 해결
표 20. 내보낸 ONNX 그래프와 대조 결과
표 21. TensorRT 정밀도 조건별 상실률·메모리·지연
표 22. 배포 선택지 넷의 비교
표 23. Orin Nano에서의 관행 빌드 시도와 실패 지점
표 24. 실행 경로별 상실률과 초과 비율
표 25. Orin Nano 실측 결과
표 26. headroom_guard의 명령
표 27. 초과 비율과 실제 상실률
표 28. 도구의 세 판정
표 29. 모델 계열별 도구 적용 결과
표 30. 학습된 λ 체크포인트의 여유와 실제 상실
표 31. 배포 조건 정확도의 종단 검증
표 32. fp16과 bf16의 짝지은 정확도 비교
표 33. 기여 진술의 부분별 근거와 강도

---

# 그림 차례

그림 1. 비전 인코더 블록별 최대 활성의 크기
그림 2. 입력 해상도에 따른 (a) 마지막 블록 활성 크기와 (b) fp16 생성 상실률
그림 3. 마지막 블록 fp16 여유와 INT8 양자화 SNR의 역전
그림 4. 정렬 손실 가중치 λ_align에 따른 (a) 블록별 활성 크기와 (b) 마지막 블록 fp16 여유
그림 5. 실행 경로별 fp16 상실률과 엣지 보드 지연
그림 6. 최악 이미지 여유에 따른 판정 기준 보정
