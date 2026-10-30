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
| TensorRT 측정 (A100) | TensorRT 10.13, 형식 비용은 10.16에서도 측정 |
| Gemma 4·Qwen3-VL·GLM-4.1V 측정 | PyTorch 2.11.0+cu128, `transformers` 5.20.2 (Gemma 4는 `transformers` 5.5 이상, Qwen3-VL은 4.49.0보다 새 판이 필요하다) |
| 엣지 보드 | Jetson Orin Nano 8GB (Ampere GPU 1,024코어, DLA 없음), L4T R36.4.7, TensorRT 10.3.0, 15W 모드 |

Gemma 4·Qwen3-VL·GLM-4.1V는 `transformers` 5.20.2에서 측정하였다. 같은 모델(Qwen2.5-VL-3B)과 같은 이미지 목록을 두 버전에서
측정하면 활성 p50은 68,608, 여유 p50은 0.9548배로 두 버전에서 같고, 최악 활성만 89,088 대 88,064로 1.15% 다르다(원본 해상도, 799장, bf16).
p50은 버전을 넘어 비교할 수 있고, 최악값은 약 1% 다르므로 버전이 다른 모델끼리 최악값을 비교하지 않는다.
Pixtral-12B와 Kimi-VL은 주 환경(4.49.0)에서 측정하였다.

## B. 엣지 보드의 측정 절차

엔진은 16번 블록을 경계로 둘로 나누어 빌드하고, 측정 시 두 분할 엔진을 차례로 적재한다. 지연은 두 분할 엔진의 합이다. 입력 전처리가 A100과
비트 단위로 같은지는 해상도별 기준 해시로 확인하였다.

## C. 공개 저장소 실측 상세

5.6절의 공개 양자화 저장소 조사에서 측정한 저장소별 값이다.

위험 구성 27개를 799장, 원본 해상도, 마지막 블록, bf16으로 측정하였다. 고정 해상도 인코더(Phi·LLaVA·InternVL3)에는 해상도 설정이 적용되지 않는다.

**<부록 표 2>** 위험 구성 공개 저장소의 실측 여유

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


(799장, 원본 해상도, 마지막 블록, bf16, 여유는 p50·최악. 판정은 3.4절의 도구 기준이다. 초과 비율이 0보다 크면 위험, 초과가 0건이고 최악 여유가 1.25배 미만이면 경계, 그 밖은
안전이다. 나머지 재양자화본 7개는 같은 기반의 표 안 저장소와 p50·최악이 같다. 해당 저장소는
`p-christ/Qwen2-VL-7B-Instruct-AWQ`, `CYF200127/Qwen3-VL-32B-Instruct-AWQ`, `remodlai`·`xxxSHADOWxxx`·`ticoAg`의
`Qwen3-VL-30B-A3B-Instruct-AWQ`, `gsting/Qwen3-VL-235B-A22B-Instruct-AWQ`, `lurker18/Phi_3.5_Vision_Instruct_AWQ_4bit`이다. Qwen3-VL은 `transformers` 5.20.2 측정이라 최악값은 다른 행과 직접 비교하지 않는다.)

5.6절의 여유 상속은 고정 해상도 인코더 세 계열에서도 확인되었다. LLaVA-1.5-7B는 기반과 AWQ가 여유 p50 319.53배·최악 291.13배로 같고 비전 가중치 텐서 391개가
모두 비트 단위로 동일하다. Phi-3.5-vision은 282.34배·최악 244.42배로 같되 bf16→fp16 재저장이라 가중치가
비트 단위로는 다르다(최대 차이 2.98×10⁻⁸). InternVL3는 기반(`SenseNova-SI-1.1-InternVL3-8B`)과 AWQ가
419.90배·최악 385.32배로 같고, 비전 가중치는 fp16 변환 뒤 340개가 모두 비트 단위로 동일하다. 위험 구성의 재양자화본 7개(Qwen2-VL 1개, Qwen3-VL 5개, Phi-3.5 1개)도 모두 같은 기반의 측정값과 p50·최악이 같다.

Phi는 가중치가 비트 단위로 같지 않은데도 여유가 같다. 재저장 반올림이 활성 최대에 영향을
미치지 않기 때문이다. 세 계열 모두 여유가 수백 배인 고정 해상도 인코더이므로, 확인한 것은 **상속**이지 위험이
아니다.

