# Jetson Orin Nano 검증 절차 (2026-09-14)

> **대상 보드 사양 확인 (2026-09-18).** Jetson Orin Nano 8GB는
> **1024-core Ampere GPU + 32 Tensor Core**이고 **DLA가 없다**(4GB는 512-core / 16).
> DLA는 상위 모듈에만 들어간다 — **AGX Orin 2× / Orin NX 1×**.
>
> 그래서 **"DLA가 bf16을 지원하지 않는다"는 논거는 이 보드에 해당하지 않는다.**
> 여기서는 GPU만 쓰므로 bf16도 선택지에 있고, fp16을 쓰는 이유는 **정밀도
> (가수 10 대 7비트)와 현재 런타임의 속도 차이(1.34~1.38배)**다. DLA 논거는
> 상위 Orin으로 확장할 때만 쓴다(§8 한계 ③).
> 출처: https://developer.nvidia.com/blog/solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/

본 연구의 가장 큰 구멍은 **"배포를 주장하는데 배포한 적이 없다"**는 점이다. 모든 측정이
A100 위 PyTorch에서 이뤄졌고, "차량용 가속기가 fp16만 지원한다"는 서술은 측정이 아니라
인용이다. 이 문서는 그 구멍을 메우는 최소 경로를 정리한다.

---

## 0-. 위상 변경 (2026-09-14)

**이 검증은 논문의 7장이 되었다.** 이전까지는 "있으면 좋은 보강"이었으나,
Orin Nano 배포를 학위논문에 포함하기로 하면서 **필수 항목**이 되었다.
장 구성·해석 사전 등록·산출물 목록은 `thesis_outline_20260910.md` 7장을 따른다.
이 문서는 그 장의 **실행 절차**다.

**논문이 이 장에서 요구하는 산출물** — 절차를 밟을 때 이 넷이 나오는지 확인할 것.

| # | 산출물 | 형태 | 대응 절 |
|---|---|---|---|
| 1 | 정밀도 조건별 붕괴율 (baseline_v2 대 full) | 표 | 7.2 |
| 2 | TensorRT 레이어별 실제 정밀도 | 빌드 로그 발췌 | 7.2 |
| 3 | fixA 적용 엔진의 붕괴율 | 표 1행 추가 | 7.4 |
| 4 | 해상도 × (붕괴율·지연시간·피크 메모리·여유) | 표 + 그림 | 7.5 |

**정밀도 조건을 반드시 둘로 나눈다.** `trtexec --fp16`은 fp16을 *허용*하는 것이지
강제하지 않으며, TensorRT는 레이어별로 fp32를 남길 수 있다. 이것을 구분하지 않으면
"재현 안 됨"의 원인을 알 수 없다.

```
# 관행 — 실제 배포에서 무슨 일이 일어나는가
trtexec --onnx=tower_baseline_v2_native.onnx --saveEngine=b_fp16.plan --fp16 --verbose

# 엄격 — 우리 주장의 수치적 조건이 성립하는가
trtexec --onnx=tower_baseline_v2_native.onnx --saveEngine=b_strict.plan \
        --fp16 --precisionConstraints=obey --noTF32 --verbose

# 대조군 — 붕괴가 정밀도 때문임을 확인
trtexec --onnx=tower_baseline_v2_native.onnx --saveEngine=b_fp32.plan
```

`--verbose` 로그에서 마지막 블록 `down_proj`가 실제 어떤 정밀도로 배치됐는지 찾아
논문에 싣는다. **"관행 조건에서 안전"이 나와도 숨기지 않는다** — 그 결과는 처방을
약화시키지 않고 정밀화한다(같은 가중치가 런타임 설정에 따라 갈리므로 여유 측정의
필요성이 오히려 커진다).

---

## 0. 목표를 둘로 나눈다

| | 범위 | 소요 | 논문 기여 |
|---|---|---|---|
| **A. 최소 검증 (권장)** | **비전 인코더만** TensorRT fp16으로 올려 붕괴 재현 여부 확인 | 반나절~하루 | **핵심 구멍을 메운다** |
| B. 전체 배포 | VLM 전체(INT4 LLM + 비전 인코더) 추론 파이프라인 | 수일 | 부가적 |

**A로 충분하다.** 검증하려는 명제가 "fp16 연산 경로에서 비전 인코더 마지막 블록이
표현 범위를 넘는가"이므로 LLM은 필요 없다. 전체 배포는 별개의 엔지니어링 과제다.

**메모리 타당성**: Qwen2.5-VL-3B 비전 인코더는 depth 32 / hidden 1280 / intermediate 3420,
약 **630M 파라미터 = fp16 기준 1.17 GiB**다. Orin Nano 8GB에 여유 있게 올라간다.
(전체 3B 모델은 fp16으로 약 6GB라 8GB 보드에서 활성·KV까지 감당하기 어렵다 —
전체 배포 시 INT4 가중치 양자화가 선택이 아니라 필수인 이유다.)

---

## 1. 사전 확인 (기기에서)

```bash
# JetPack / TensorRT / CUDA 버전
cat /etc/nv_tegra_release
dpkg -l | grep -E "nvidia-l4t-core|tensorrt"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

확인할 것:
- **TensorRT가 bf16을 지원하는가.** 지원하면 "비전 인코더만 bf16 유지"라는 대응책이
  생긴다. Ampere 아키텍처 자체는 bf16을 지원하나 Jetson의 TensorRT 경로 지원은
  버전에 따라 달랐다 — **반드시 실측할 것.** (`trtexec --bf16` 수용 여부)
- 사용 가능한 메모리 (`free -h`), 전력 모드 (`nvpmodel -q`)

---

## 2. A 절차 — 비전 인코더만 검증

### 2-1. 비전 인코더를 ONNX로 내보낸다 (x86 개발 머신에서)

해상도를 **고정**한다. Qwen2.5-VL 비전 인코더는 `grid_thw`로 회전 위치 임베딩과 윈도우
인덱스를 계산하므로 동적 형상이면 export가 어렵다. 배포 해상도를 정하고 그 값으로 고정하면
정적 형상이 되어 export가 가능해진다.

```python
# 개념 코드 — 실제로는 grid_thw를 상수로 굽는 래퍼가 필요하다
class TowerWrapper(torch.nn.Module):
    def __init__(self, tower, grid_thw):
        super().__init__(); self.tower = tower
        self.register_buffer("grid", grid_thw)
    def forward(self, pixel_values):
        return self.tower(pixel_values, self.grid)

torch.onnx.export(TowerWrapper(visual, grid), dummy_pixels,
                  "vision_tower.onnx", opset_version=17,
                  input_names=["pixel_values"], output_names=["image_embeds"])
```

**주의**: 이 export는 한 번에 되지 않을 수 있다. 윈도우 attention·`cu_seqlens` 계산에
파이썬 제어 흐름이 들어가므로, 고정 해상도 전제로 해당 부분을 상수화하는 패치가
필요할 수 있다. 여기서 막히면 **2-4의 대체 경로**로 간다.

### 2-2. TensorRT 엔진 빌드 (Orin에서)

```bash
trtexec --onnx=vision_tower.onnx --saveEngine=tower_fp16.plan --fp16
trtexec --onnx=vision_tower.onnx --saveEngine=tower_fp32.plan          # 대조군
# bf16을 지원하면
trtexec --onnx=vision_tower.onnx --saveEngine=tower_bf16.plan --bf16
```

### 2-2b. 내보내 둔 파일 (2026-09-14 완료)

네 개 모두 `scripts/export_vision_tower_onnx.py`로 만들었고 검증을 통과했다.
`.onnx`(그래프)와 `.onnx.data`(가중치)를 **같은 디렉터리에 함께** 옮겨야 한다.

| 파일 | `max_pixels` | 패치 | 상수화 대조 | ONNX 코사인 | 최대 절대오차 | 크기 |
|---|---|---|---|---|---|---|
| `onnx/tower_baseline_v2_native.onnx` | 1,440,000 | 7,296 | 0.000e+00 | 0.999920 | 8.847 | 2.69 GiB |
| `onnx/tower_full_native.onnx` | 1,440,000 | 7,296 | 0.000e+00 | 0.999999 | 0.358 | 2.69 GiB |
| `onnx/tower_baseline_v2.onnx` | 200,704 | 924 | 0.000e+00 | 0.999978 | 0.839 | 2.63 GiB |
| `onnx/tower_full.onnx` | 200,704 | 924 | 0.000e+00 | 1.000000 | 0.111 | 2.63 GiB |

**주 대조쌍은 `*_native` 둘**이다(A100 기준값 94.8% 대 0.0%). 200,704 판본은
Orin에서 원본 해상도가 OOM일 때의 대비책이자 파이프라인 점검용이다(A100 기준값
3.2% 대 0.0%로 판정력은 약하다). 200,704 판본 두 개는 dedup 도입 전에 만들어져
중복 상수가 0.14 GiB씩 남아 있으나 동작에는 영향이 없다.

**부수 관찰**: 두 해상도 모두에서 `student_full`의 코사인이 더 높고 절대오차가 작다
(원본 해상도 0.358 대 8.847, 약 25배). 활성 크기가 작으면 fp32 누적 오차도 작다는
뜻으로, 본 논문의 논지와 방향이 일치한다.

### 2-3. 같은 250장으로 붕괴율 측정

`eval_results/resolution_sweep/` 과 **동일한 이미지·동일한 해상도**를 써야 A100 결과와
직접 비교된다. 각 이미지에 대해 출력에 NaN/Inf가 있는지만 세면 된다.

```
붕괴율 = NaN이 발생한 이미지 수 / 250
```

대상 체크포인트는 최소 둘: `student_baseline_v2`(A100 fp16에서 94.8% 붕괴)와
`student_full`(0%). **둘의 차이가 재현되는지**가 핵심이다.

**어느 해상도로 잴 것인가** — A100 실측(`eval_results/resolution_sweep/`, 250장, fp16
마지막 블록 붕괴율)은 다음과 같다. 해상도가 낮아질수록 두 체크포인트의 차이가 좁아지므로,
**Orin에서 실행 가능한 가장 높은 해상도**를 골라야 판정력이 남는다.

| `max_pixels` | 패치 수 | `student_baseline_v2` | `student_full` | 차이 |
|---|---|---|---|---|
| 1,440,000 (원본) | 7,296 | **94.8%** | 0.0% | 94.8pp |
| 802,816 | 4,096 | 25.6% | 0.0% | 25.6pp |
| 401,408 | 2,048 | 10.4% | 0.0% | 10.4pp |
| 200,704 (평가 설정) | 1,024 | 3.2% | 0.0% | 3.2pp |
| 100,352 | 512 | 0.8% | 0.0% | 0.8pp |

**메모리 제약이 해상도 상한을 정한다.** export한 그래프는 HF 원본과 같은 eager attention
이라 어텐션 행렬 `(head=16, S, S)`를 통째로 만든다. 원본 해상도(S=7,296)에서 이것만
fp16 1.59 GiB이고 softmax가 fp32로 승격되면 3.17 GiB가 더 붙는다. 타워 가중치 1.17 GiB와
윈도우 마스크 상수 0.2 GiB를 더하면 피크가 **6 GiB 안팎**이라, 8GB를 OS와 공유하는
Orin Nano에서는 아슬아슬하다(TensorRT가 softmax까지 fp16으로 유지하면 4.6 GiB 수준).
원본 해상도에서 OOM이 나면 802,816(S=4,096, 어텐션 fp32 1.0 GiB)으로 물러선다 —
25.6% 대 0.0%도 충분히 갈리는 차이다.

그래도 원본 해상도가 필요하면 **창(window) 어텐션을 실제로 창 단위로 재구성**하는 방법이
있다. 32개 블록 중 28개는 윈도우 어텐션이라 `(창 120개, 16, 64, 64)`로 배치화하면
어텐션 메모리가 3.17 GiB → **30 MiB**로 떨어진다. 지금 export 스크립트는 원본과의 수치
일치를 우선해 이 최적화를 넣지 않았다(원본 HF 구현도 밀집 행렬을 만든다).

**export 시 주의 — 트레이서가 마스크를 블록마다 복제한다 (2026-09-14 실측)**
`torch.onnx.export`(torch 2.3, `torch.jit.trace` 경로)는 입력과 무관한 큰 텐서를
**소비 노드마다 별개의 `Constant` 노드로** 내보낸다. 원본 해상도에서 마스크 하나가
203 MiB이고 윈도우 어텐션 블록이 28개라, 파일이 2.49 GiB(가중치) + 5.83 GiB(중복 마스크)
= **8.32 GiB**로 부풀었다.

이 부풀기는 **초기화값만 세면 보이지 않는다** — 중복된 상수는 그래프 initializer가 아니라
노드 attribute로 들어가고, `onnx.save_model(convert_attribute=True)`가 그것까지 외부
데이터로 쓰기 때문이다.

**시도해서 실패한 두 가지**(익스포터 내부 동작을 추정해 우회하려던 것):
1. `do_constant_folding=False` — 원본 해상도에서 5.83 GiB 그대로.
2. 마스크를 모듈 속성 대신 `cu_seqlens` 인자(그래프 값)로 전달 — 저해상도에서 91 MiB 그대로.

**실제 해법**: 저장 직전에 결과 그래프를 직접 고친다. `dedup_constants()`가 내용이 같은
큰 `Constant` 노드를 (shape, dtype, SHA-1) 키로 묶어 **공유 초기화값 하나**로 대체하고
참조 노드들의 입력을 그쪽으로 돌린다. 익스포터 버전에 의존하지 않는다는 이점도 있다.
저해상도 검증: 28개 → 1개, 91 MiB → 0 MiB, 파일 2.62 → 2.53 GiB, ONNX 런타임
코사인 중앙값 0.999970으로 의미 보존 확인.

### 2-4. 대체 경로 (ONNX export가 막힐 경우)

Jetson용 PyTorch로 비전 인코더를 직접 fp16 실행한다. TensorRT의 융합 정책은 검증하지
못하지만 **실제 하드웨어에서의 fp16 산술**과 메모리 타당성은 확인된다.
`scripts/profile_vision_activations.py`를 그대로 쓸 수 있다(모델 로드 경로만 조정).

---

## 3. 판정과 해석

| 결과 | 해석 | 논문 반영 |
|---|---|---|
| **fp16에서 붕괴 재현** (baseline 높음 / full 0%) | A100 결과가 실제 배포 스택에서 성립 | **가장 강한 결과.** "배포 미검증" 한계가 사라진다 |
| **fp16에서 붕괴 안 남** | TensorRT가 MLP 내부 누적을 fp32로 유지하는 등 융합 정책이 다름 | 그것대로 중요한 발견 — **"프레임워크에 따라 다르므로 배포 스택에서 직접 재라"**는 처방이 강화된다 |
| bf16 미지원 확인 | "bf16으로 내보내면 되지 않나"라는 반론에 **실측 근거**가 생긴다 | §6 예상 질문 보강 |

**어느 결과든 논문에 쓸 수 있다.** 두 번째 경우에도 논지("측정하라")는 오히려 강해진다.

---

## 4. 추가로 값싸게 얻을 수 있는 것

Orin에서 재는 김에 함께 측정하면 좋은 것:

- **지연시간·메모리** — 현재 논문에 경량화를 표방하면서 효율 수치가 하나도 없다.
  비전 인코더만이라도 해상도별 지연시간을 재면 3장에 표 하나가 채워진다.
- **해상도 ↔ 여유 트레이드오프 곡선** — Orin에서 실행 가능한 해상도 상한과, 그 해상도에서의
  fp16 여유를 함께 표시하면 **"이 기기에서 안전한 최대 해상도"**라는 배포 파라미터가 나온다.
  이것이 자동차공학회 청중에게 가장 직접적인 산출물이다.

---

## 5. B (전체 배포) 개요 — 참고

당장 필요하지 않으나 기록해 둔다.

1. LoRA 병합 → INT4 양자화 (AWQ 또는 GPTQ). 8GB 보드에서는 필수
2. 추론 런타임 선택 — TensorRT-LLM(Jetson 지원 확인 필요), llama.cpp 계열, MLC 등
3. 비전 인코더와 LLM의 **연산 dtype이 서로 다를 수 있음**에 주의.
   본 연구의 문제가 정확히 이 지점에서 발생했다(LLM은 INT4, 비전 인코더는 저장 dtype)
4. 프롬프트·이미지 전처리를 학습 시 설정과 일치시킬 것 (`min/max_pixels`)

---

## 6. 이 검증이 논문에 채우는 것

`thesis_outline_20260910.md` 7장 한계의 첫 항목:

> **다른 VLM 계열에서 손실 설계 축을 검증하지 못함** / 실배포 스택 미검증

이 중 후자가 해소된다. 그리고 §6 예상 질문의 "그냥 bf16 쓰면 되지 않나"에
**실측 근거**가 붙는다.
