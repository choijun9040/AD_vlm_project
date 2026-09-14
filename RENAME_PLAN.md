# `L_spatial` → `L_align`, `L_temporal` → `L_atc` 명칭 통일 계획

명칭 결정 근거는 `paper_positioning_ksae2026.md` §2.

## 완료 (2026-09-11)

동작에 영향이 없는 주석·문서 문자열만 먼저 치환했다.

| 파일 | 치환 |
|---|---|
| `scripts/dataloader.py` | 4곳 |
| `scripts/train_baseline.py` | 2곳 |
| `scripts/train_kd_only.py` | 2곳 |
| `scripts/probe_vision_count.py` | 2곳 |

`scripts/plot_loss_curves.py`는 이미 표시명을 새 이름으로 쓰고 있다(로그 키만 구명 파싱).

## 완료 (2026-09-14) — `scripts/train_distillation.py`

λ 스윕이 2026-09-12에 끝나 안전해진 뒤 적용했다. 적용 시점에 B(정렬 대상 ablation)가
실행 중이었으나, Python이 모듈을 이미 메모리에 올린 뒤라 실행에는 영향이 없었다.

**바꾼 것**: 클래스명(`SpatialFeatureKDLoss`→`FeatureAlignKDLoss`,
`TemporalContextKDLoss`→`AsymmetricTemporalKDLoss`), 변수명(`*_criterion`, `loss_*`),
CONFIG 키(`lambda_align` / `lambda_atc` / `align_layer_idx`), CLI 플래그,
로그 출력 키(`align=` / `atc=`), 주석·docstring.

**하위 호환**:
- 구 클래스명을 **모듈 수준 alias**로 남겼다(같은 객체이므로 구 이름 monkeypatch도 동작).
- `_normalize_config()`가 구 CONFIG 키를 신 키로 옮긴다(진입 시 로그 출력).
- 구 CLI 플래그 `--lambda_spatial` / `--lambda_temporal`을 deprecated alias로 유지.

**유지한 것** (계획대로): `--variant spatial`/`temporal`, `spatial_proj.pt` /
`temporal_proj.pt` 파일명, `checkpoints/student_spatial` 등 출력 경로.
기존 산출물·재개 경로와의 연결을 끊지 않기 위함이다.

**함께 고친 것**: `scripts/plot_loss_curves.py`의 `COMP_RE`가 구·신 로그 키를 모두 받는다
(`task|spatial|align|temporal|atc|kd`). 이미 쌓인 로그는 구 키다.
`scripts/train_align_target_ablation.py`의 monkeypatch 대상명·CONFIG 키 참조도 갱신했다.

**검증**: 구문 검사, import 및 alias 동일 객체 확인, 구 키 정규화 동작 확인,
미정의 이름 정적 검사(잔여 없음), 로그 출력 키·`epoch_losses` 키 일치 확인.
백업: `/root/.claude/jobs/f67d7bab/tmp/train_distillation.py.bak`

---

## (참고) 원래 계획 — λ 스윕 종료 후

`scripts/train_distillation.py`는 **지금 건드리지 않는다.**

이유: `_orchestrate_lambda_sweep.sh`가 λ=1.0, λ=2.0 구간을 아직 실행하지 않았고
`--variant spatial --lambda_spatial`을 호출한다. 지금 수정하면

1. CLI 플래그·변형명을 바꿀 경우 남은 두 구간이 **실행 실패**하고,
2. 플래그를 유지하더라도 **4개 구간이 서로 다른 코드로 학습**되어
   "λ만 다르다"는 통제 실험의 전제가 흔들린다. 심사에서 "네 실행의 코드가
   동일했는가"에 "아니오"로 답해야 한다.

스윕이 끝난 뒤 아래를 한 번에 적용한다.

### 바꿀 것

| 대상 | 현재 | 변경 후 |
|---|---|---|
| 클래스 | `SpatialFeatureKDLoss` | `FeatureAlignKDLoss` |
| 클래스 | `TemporalContextKDLoss` | `AsymmetricTemporalKDLoss` |
| 변수 | `spatial_criterion` / `temporal_criterion` | `align_criterion` / `atc_criterion` |
| 변수 | `loss_spatial` / `loss_temporal` | `loss_align` / `loss_atc` |
| CONFIG 키 | `lambda_spatial` / `lambda_temporal` | `lambda_align` / `lambda_atc` |
| CONFIG 키 | `spatial_layer_idx` | `align_layer_idx` |
| CLI 플래그 | `--lambda_spatial` / `--lambda_temporal` | `--lambda_align` / `--lambda_atc` |
| 로그 출력 | `spatial=` / `temporal=` | `align=` / `atc=` |
| 주석·docstring | `L_spatial` / `L_temporal` | `L_align` / `L_atc` |

### 유지할 것 (호환성)

- **`--variant spatial` / `temporal`** — `checkpoints/student_spatial`,
  `student_temporal` 및 `eval_results/student_spatial_*.json` 등 이미 생성된
  산출물 경로와 1:1 대응한다. 바꾸면 기존 결과 파일과의 연결이 끊긴다.
- **`spatial_proj.pt` / `temporal_proj.pt`** — 기존 모든 체크포인트가 이 이름으로
  저장되어 있고 `resume_from`이 읽는다. 바꾸면 재개가 깨진다.
- **구 CLI 플래그를 alias로 남긴다** (`--lambda_spatial`을 `--lambda_align`의
  deprecated alias로). 기존 오케스트레이션 스크립트가 계속 동작하도록.

### 함께 고칠 것

- `scripts/plot_loss_curves.py`의 `COMP_RE`가 **구·신 키를 모두** 받도록 수정:
  `(task|spatial|align|temporal|atc|kd)=`. 이미 쌓인 로그(`full.log`, `spatial.log`,
  `temporal.log`, `sweep_align*.log`)는 구 키로 남아 있으므로 양쪽을 읽어야 한다.
- `_orchestrate_lambda_sweep.sh` 주석의 `lambda_spatial` 표기.

### 검증 절차

1. `--max_steps 2`로 스모크 실행해 체크포인트 저장·로그 출력 확인
2. 기존 체크포인트에서 `resume_from`으로 1스텝 재개 확인 (proj 로딩)
3. `plot_loss_curves.py`로 구 로그와 신 로그 양쪽 파싱 확인
