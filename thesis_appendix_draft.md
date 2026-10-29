<!-- 부록 초안 (2026-10-22 신설). 본문을 결과 중심으로 고치며(개요 §6 「경과 기록의 처리」·「결과 중심 작성」) 재현에 필요한
     측정 환경과 보드 측정 절차를 본문에서 이곳으로 옮겼다(빌드 실패 기록은 같은 날 논문에서 뺐다 — 개요 §6.4). scripts/build_thesis.py가 참고 문헌 뒤·감사의 글 앞에 넣는다.
     소제목은 `###`로 둔다 — `##`는 본문 절로 세어져 합본의 절 수 대조가 깨진다. -->

---

# 부록

## A. 측정 환경

측정에 쓴 소프트웨어와 하드웨어는 <부록 표 1>과 같다.

**<부록 표 1>** 측정 환경

| 용도 | 환경 |
|---|---|
| 학습·PyTorch 측정 (주 환경) | Python 3.10, PyTorch 2.3.0, `transformers` 4.49.0, `peft` 0.14.0, NVIDIA A100 80GB |
| TensorRT 측정 (A100) | TensorRT 10.13, 형식 비용은 10.16에서도 측정. 주 환경과 분리한 가상환경 |
| Gemma 4·Qwen3-VL·GLM-4.1V 측정 | PyTorch 2.11.0+cu128, `transformers` 5.20.2 (Gemma 4는 5.5 이상, Qwen3-VL은 4.49.0보다 새 판이 필요하다) |
| 엣지 보드 | Jetson Orin Nano 8GB (Ampere GPU 1,024코어, DLA 없음), L4T R36.4.7, TensorRT 10.3.0, 15W 모드 |

Qwen3-VL은 주 환경의 `transformers` 4.49.0이 아키텍처(`qwen3_vl`)를 인식하지 못해 새 버전에서만
적재된다. 본 논문에 두 버전의 수치를 섞지 않도록, 같은 모델(Qwen2.5-VL-3B)과 같은 이미지 목록을 두 버전에서 측정하였다(<부록 표 2>).

**<부록 표 2>** `transformers` 버전에 따른 같은 모델의 측정값 (원본 해상도, 799장, bf16)

| 항목 | 4.49.0 | 5.20.2 | 차이 |
|---|---|---|---|
| 활성 p50 | 68,608 | 68,608 | 0.0000% |
| 여유 p50 | 0.9548 | 0.9548 | 0.0000% |
| 활성 최악 | 89,088 | 88,064 | −1.15% |


p50은 버전을 넘어 비교할 수 있고, 최악값은 약 1% 다르므로 버전이 다른 모델끼리 최악값을 비교하지 않는다.
한편 Pixtral-12B와 Kimi-VL은 주 환경(4.49.0)에서 측정하였다.

## B. 엣지 보드의 측정 절차

메모리 때문에 페이지 캐시를 비운 상태에서, 빌드한 엔진을 적재하지 않고 파일로만 저장하는 빌더로 엔진을
만들고, 측정은 16번 블록을 경계로 나눈 두 분할 엔진을 차례로 적재하여 수행한다. 지연은 두 분할 엔진의 합이다. 입력 전처리가 A100과
비트 단위로 같은지는 해상도별 기준 해시로 확인하였다.

## C. 공개 저장소 실측 상세

4.6절의 공개 양자화 저장소 조사에서 측정한 저장소별 값이다.

위험 구성 27개를 799장, 원본 해상도, 마지막 블록, bf16으로 측정하였다. 고정 해상도 인코더(Phi·LLaVA·InternVL3)에는 해상도 설정이 적용되지 않는다.

**<부록 표 3>** 위험 구성 공개 저장소의 실측 여유

| 저장소 | 여유 p50 | 최악 | 판정 |
|---|---|---|---|
| **`AngelSlim/Qwen2.5-VL-3B-Instruct-INT4-AWQ`** | **0.95배** | 0.74배 | **위험** |
| **`AngelSlim/Qwen2.5-VL-72B-Instruct-AWQ`** | **1.13배** | 0.88배 | **위험** |
| `AngelSlim/Qwen2.5-VL-7B-Instruct-AWQ` | 1.73배 | 1.27배 | 안전 |
| `AngelSlim/Qwen2.5-VL-32B-Instruct-AWQ` | 1.97배 | 1.38배 | 안전 |
| `Qwen/Qwen2-VL-2B-Instruct-AWQ` (공식) | 3.91배 | 3.55배 | 안전 |
| `Qwen/Qwen2-VL-7B-Instruct-AWQ` (공식) | 4.51배 | 3.88배 | 안전 |
| `QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ` | 4.97배 | 4.55배 | 안전 |
| `QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ` | 4.99배 | 4.69배 | 안전 |
| `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` | 6.09배 | 5.30배 | 안전 |
| `QuantTrio/Qwen3-VL-32B-Thinking-AWQ` | 7.42배 | 6.24배 | 안전 |
| `Qwen/Qwen2-VL-72B-Instruct-AWQ` (공식) | 10.23배 | 8.39배 | 안전 |
| `QuantTrio/Qwen3-VL-30B-A3B-Thinking-AWQ` | 40.33배 | 32.75배 | 안전 |
| `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ` | 41.99배 | 33.42배 | 안전 |
| `RichardErkhov/muhtasham_-_Phi-3.5-vision-instruct_*-awq` (3개) | 240.82 · 250.02 · 308.98배 | 216.90 · 227.44 · 257.89배 | 안전 (고정 해상도) |
| `Isotr0py/Phi-3.5-vision-instruct-AWQ` | 282.34배 | 244.42배 | 안전 (고정 해상도) |
| `lurker18/Phi_3_Vision_128k_Instruct_AWQ_4bit` | 293.74배 | 255.88배 | 안전 (고정 해상도) |
| `ybelkada/llava-1.5-7b-hf-awq` | 319.53배 | 291.13배 | 안전 (고정 해상도) |
| `ELVISIO/SenseNova-SI-InternVL3-8B-AWQ` | 419.90배 | 385.32배 | 안전 (고정 해상도) |


(판정은 6.2절의 도구 기준이다: 초과 비율이 0보다 크면 위험, 초과가 0건이고 최악 여유가 1.25 미만이면 경계, 그 밖은
안전. 나머지 재양자화본 7개는 같은 기반의 표 안 저장소와 p50·최악이 같다. 해당 저장소는
`p-christ/Qwen2-VL-7B-Instruct-AWQ`, `CYF200127/Qwen3-VL-32B-Instruct-AWQ`, `remodlai`·`xxxSHADOWxxx`·`ticoAg`의
`Qwen3-VL-30B-A3B-Instruct-AWQ`, `gsting/Qwen3-VL-235B-A22B-Instruct-AWQ`, `lurker18/Phi_3.5_Vision_Instruct_AWQ_4bit`이다.)

**<부록 표 4>** 기반 모델과 양자화 파생본의 여유 상속

| 저장소 | 활성 p50 | 여유 p50 | 최악 |
|---|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` (기반) | 68,608 | 0.95 | 0.74 |
| **`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`** (공식) | **68,608** | **0.95** | **0.74** |
| `AngelSlim/Qwen2.5-VL-3B-Instruct-INT4-AWQ` | 68,608 | 0.95 | 0.74 |
| `Qwen/Qwen2.5-VL-7B-Instruct` (기반) | 38,144 | 1.72 | 1.236 |
| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (공식) | 38,144 | 1.72 | 1.248 |
| `AngelSlim/Qwen2.5-VL-7B-Instruct-AWQ` | 37,888 | 1.73 | 1.27 |


(799장, 원본 해상도, bf16. 7B 기반과 공식 7B AWQ는 최악 여유가 1.25 아래라 도구 판정이 「경계」다)

LLaVA-1.5-7B는 기반과 AWQ가 여유 p50 319.53배·최악 291.13배로 같고 비전 가중치 텐서 391개가
모두 비트 단위로 동일하다. Phi-3.5-vision은 282.34배·최악 244.42배로 같되 bf16→fp16 재저장이라 가중치가
비트 단위로는 다르다(최대 차이 2.98×10⁻⁸). InternVL3는 기반(`SenseNova-SI-1.1-InternVL3-8B`)과 AWQ가
419.90배·최악 385.32배로 같고, 비전 가중치는 fp16 변환 뒤 340개가 모두 비트 단위로 동일하다. 위험 구성의 재양자화본 7개
(Qwen2-VL 1개, Qwen3-VL 5개, Phi-3.5 1개)도 모두 같은 기반의 측정값과 p50·최악이 같다.

Phi는 가중치가 비트 단위로 같지 않은데도 여유가 같다. 재저장 반올림이 활성 최대에 영향을
미치지 않기 때문이다. 세 계열 모두 여유가 수백 배인 고정 해상도 인코더이므로, 확인한 것은 **상속**이지 위험이
아니다.

Qwen3-VL은 `transformers` 5.20.2에서 측정하였다(부록 A: 같은 모델의 p50은 버전 간 차이가 없다). 세대별 여유는 <부록 표 5>에 정리하였다(원본 해상도, 799장, p50).

**<부록 표 5>** Qwen 세대·크기별 여유

| 세대 | 모델 | 여유 |
|---|---|---|
| Qwen2-VL | 2B · 7B · 72B (공식 AWQ) | 3.91 · 4.51 · 10.23 |
| **Qwen2.5-VL** | **3B · 7B · 32B · 72B** | **0.95 · 1.72 · 1.97 · 1.13** |
| Qwen3-VL | 32B (dense) · 30B-A3B (MoE) · 235B-A22B (MoE) | 6.09 · 41.99 · 4.99 |


(Qwen2.5-VL 3B·7B는 기반, 32B·72B는 AngelSlim AWQ. Qwen3-VL은 Instruct 판)
