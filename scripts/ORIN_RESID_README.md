# Orin Nano 측정 — (R) 잔차 재조정 교정본

사전 등록 `thesis_outline_20260910.md` §6 개정 10(2026-11-03). 6장 [orin-result]의 교정본 행을 **(R) 잔차 재조정 교정본**으로 다시 잰다.
조건은 2026-10-02·799장 측정과 같다 — 강타입 `--stronglyTyped`, fp16 그래프(블록 16 분할 + 어텐션 나눠 계산), Python 빌더, 분할 엔진 순차
측정, 15W, DriveLM 검증 CAM_FRONT 799장.

| 해상도 | 그래프 | 이전 결과 (기준선 · (P) 교정본, 799장) |
|---|---|---|
| 802,816 | `tower_baseline_v2_802816h_resid` | 기준선 30.7% · (P) 교정본 0% |
| 401,408 | `tower_baseline_v2_401408h_resid` | 기준선 · (P) 교정본 0% |

- (R) 가중치는 `checkpoints_fix/baseline_v2_residual_native.pt`(c = ¼)이고, 정규화 eps 변경(옆 파일)은 내보낼 때 그래프 상수로 들어갔다.
- 원본 해상도는 넣지 않았다(이전과 같이 보드에서 강타입으로도 빌드되지 않는다).

## 옮기기 (서버 → 보드)

```
onnx/orin_pkg_resid_<날짜>.tar.gz   스크립트 · 이미지 799장 · 전처리 캐시 · 체크섬
onnx/orin_onnx_resid_<날짜>.tar     ONNX 8개
onnx/SHA256SUMS_resid
```

보드의 같은 디렉터리에 두 묶음을 풀면 `orin_pkg/` 하나로 들어간다. `sha256sum -c SHA256SUMS_resid`로 먼저 확인한다.

## 실행 (보드)

```
cd orin_pkg
MODELS_TPL="tower_baseline_v2_{RES}_resid" bash orin_run_full799.sh            # 802816h → 401408h
```

다시 실행하면 이미 만든 엔진·결과는 건너뛴다. 끝나면 `orin_results_full799_<시각>.tar.gz`가 생긴다 — 서버로 돌려보낸다.
마지막에 A100 (R) 강타입 값(`a100_ref/trt_*_typed_resid.json`)과 나란히 출력한다.
