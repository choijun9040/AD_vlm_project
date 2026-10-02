# 학위 논문 초록 초안 (2026-09-29)

> **이 문서의 위치.** 학위 논문 국문·영문 초록 초안이다. 학술대회 요약문(`ksae2026_abstract.md`)은
> **발견**(지표 역전)을 세우는 글이고, 이 초록은 제목대로 **측정과 진단 체계** — 발견, 기전, 공개 모델에서의
> 범위, 도구, 배포 런타임 확인 — 를 요약한다. 근거는 9장 결론 초안의 9.1·9.2절이다.
>
> **작성 규칙.** 수치마다 조건(해상도·통계량)을 짧게 붙인다. 확인된 것은 A100의 TensorRT fp16 엔진까지다.
> Orin은 강타입 엔진으로 확인한 결과 한 문장을 넣었다(6.9절, 2026-10-02). 관행 빌드 불가 등 조건은 본문에 둔다.
> 분량은 국문 약 1,160자(공백 제외), 영문 약 450단어(제목·주요어 포함)다. 학교 규정 분량이 정해지면 맞춘다.
>
> **정확도 문장의 범위.** "다섯 학생이 정확도로 구별되지 않는다"로 쓰지 않는다 — 출력 증류 학생(46.74%)은
> 뚜렷이 낮다(3.6절). 구별되지 않는 것은 특징 증류 세 학생과 기준선(50.30~51.69%)이다.

---

## 국문 초록

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
1.36%(fp16)로 떨어졌다. 같은 두 학생의 INT8 양자화 SNR은 붕괴하는 쪽이 32.23 dB, 붕괴하지 않는 쪽이
22.88 dB로, **표준 압축 지표는 위험을 역방향으로 가리켰다.** 이는 양자화 도구가 검사하는 부분과 양자화에서
제외되어 fp16으로 실행되는 부분이 서로 다르기 때문이며, 조사한 공개 양자화 저장소 197개 중 13.7%가 이미
그런 설정이었다(배포 시 fp16 변환을 세지 않은 하한).

오버플로 지점은 마지막 블록 MLP의 출력 투영 안, 사전학습 가중치에 고정된 한 채널로 특정되었다. 곱이
만들어지기 전에 입력을 줄이면 학습 없이 붕괴가 사라졌고, 출력을 줄이면 사라지지 않았다. 여유는 특징 정렬
손실의 가중치에 따라 단조롭게 변했고, 정렬 대상을 학생 자신으로 바꿔도 효과의 절반이 남았으며, 저장소
설정·인코더 설계·가중치·모델 계열 어느 것으로도 예측되지 않았다. 비전 인코더를 양자화에서 뺀 공개 파생본은
기반 모델의 여유를 그대로 물려받았다.

이 진단을 도구 `headroom_guard`로 구현했다. 도구는 활성 크기를 넘치지 않는 형식(bf16)에서 재고, 형식 상한을
넘는 이미지의 비율로 붕괴율을 추정하며(PyTorch fp16 기준 6개 모델 평균 오차 2.1%p), 대상이 지원하는 형식에 맞춰 형식 전환이나
가중치 교정을 권고한다. 배포 조건에서 도구가 교정한 가중치는 정확도를 1.36%에서 50.73%로 되돌렸다. 비전
인코더를 ONNX로 내보내 TensorRT fp16 엔진으로 실행해도 같은 붕괴가 재현되었다. 엣지 보드(Jetson Orin Nano
8GB)의 강타입 엔진도 A100과 같은 비율로 붕괴했고(802,816 해상도, 24.4%), 교정 엔진은 붕괴하지 않았다.

압축된 주행 VLM의 저정밀 배포에서 부족한 것은 여유를 움직이는 수단이 아니라 **그것을 재는 일**이다.

**주요어**: 표현 범위 여유, 저정밀 추론, 지식 증류, 비전-언어 모델, 자율주행, 배포 검증

---

## Abstract

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
**the standard compression metric ranks the risk in reverse.** This happens because the part that quantization
tools inspect differs from the part that is excluded from quantization and runs in fp16 — a configuration
already declared by 13.7% of 197 public quantized repositories we surveyed, a lower bound that ignores fp16
conversion at deployment.

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
