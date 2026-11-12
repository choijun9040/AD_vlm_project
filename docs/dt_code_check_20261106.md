# Draw Things FLUX.1 fp16 처리 — draw-things-community Flux1.swift 대조 (2026-11-06 확인)

## 출처·버전
- draw-things-community `main` = **b002b883ed6904fcb7ee815db8a4fcc52dccacb3**. Flux1.swift를 마지막으로 바꾼 커밋 = **060f99d6c62b4dfe79e632aa37f4d8dfd416fdd8** (2026-10-12). 파일 2,184줄.
  https://raw.githubusercontent.com/drawthingsai/draw-things-community/main/Libraries/SwiftDiffusion/Sources/Models/Flux1.swift
- 블로그(engineering.drawthings.ai, 2025-04-30)가 건 `Flux1.swift#L497`: 블로그 시점의 마지막 Flux1.swift 커밋 **0054f779b96d9b28af1d5ee14975cd991c5987c6** (2025-04-02)의 497행이 정확히
  `hw: h * w, contextBlockPreOnly: false, upcast: i > (layers.0 - 3),` — 즉 링크 대상은 블록 17·18 조건 줄. 현재 main에서는 578행으로 옮겨졌고 내용은 같다.
- swift-diffusion **1230c66ebf8bbb63c576a29c7cab91ff8491ec3a** (커밋 시각 2026-10-19) `examples/flux/main.swift`(827줄) — 같은 논리(아래 비교).
- 귀속: draw-things-community README 5행 "the source code for our re-implementation of image generation models, samplers, data models, trainer **within the app**".
  README에는 SwiftDiffusion = liuliu/swift-diffusion이라는 명시가 **없다**. 다만 Flux1.swift 커밋 이력의 작성자는 전부 `liuliu`(swift-diffusion 소유자, Draw Things 개발자).
  → 「Draw Things 구현」의 1차 근거는 swift-diffusion이 아니라 **draw-things-community Flux1.swift 자체**로 인용하는 것이 맞다(README가 앱 내부 구현이라고 밝힘).
  swift-diffusion examples는 같은 작성자의 예제/원형으로만 부기.

## 블로그 본문(FLUX.1)
- "Activation scaling factor: 8 / Scaled layers: Double stream blocks 17, 18 (0-indexed)" + 위 #L497 링크. Pre-scaling(어텐션) 항목은 FLUX.1에는 **없음**(Wan·HiDream에만).
- 일반 서술: "we upcast the main activation accumulation path to FP32", 블록 진입 adaLN 경계에서 FP16으로, "we upcast to FP32 after the final GEMM in the MLP and perform the adaptive layernorm gating in float32".

## 항목별 대조 (Flux1.swift main 행 번호 / 우리 scripts/cmp_flux.py)
1. **잔차 fp32 — 일치.** 503/536 `out = xEmbedder(x)...to(.Float32)`, 560 `context = embedder(contextIn).to(.Float32)`.
   포트: x_embedder·context_embedder 출력 `.float()` 훅(cmp_flux.py 43행). 블록 진입은 `xNorm1(x).to(.Float16) .* xChunks[1] + xChunks[0]`(173·160행) — 포트 norm1/norm1_context/norm2/norm2_context `.half()` 훅(45–46행)과 일치. 변조 청크는 fp16(포트: tr.half()라 fp16).
2. **블록 17·18, ×8 — 일치.** 메인 `Flux1`: 578행 `upcast: i > (layers.0 - 3)` (FLUX.1 layers.0=19 → i ≥ 17). LoRA 변형 `LoRAFlux1`: 1187행 `upcast: i > 16` (리터럴).
   swift-diffusion 1230c66e 277행 `upcast: i > 16)  // Just last layer should be enough, but to be safe, for the last 2 layers, we will do upcast.`
   이미지 FF(275행)·텍스트 FF(256행) 둘 다 `upcast: upcast`. 포트: DT_BLOCKS=(17,18), ff.net[2]·ff_context.net[2] 둘 다(47–50행). 일치.
3. **FF 출력 fp32 ×8 — 일치.** FeedForward 130–146행: `out = (1 / scaleFactor) * out`(GELU 뒤, 하향 GEMM 앞) → `outProjection(out)` → `out = out.to(.Float32) * scaleFactor`.
   포트: net[2] pre-hook ×1/8, forward-hook `o.float()*8`. 편향은 양쪽 모두 사실상 ×8(가중치 매퍼 300행대 `img_mlp.2.bias`를 그대로 매핑, 1/8 보정 없음) — 같은 성질.
4. **게이팅 fp32 — 일치.** 277–279행 `xOut = xOut + xFF(...) .* xChunks[5].to(of: xOut)` (게이트를 fp32로 올려 fp32 곱), 텍스트 258–262행 동일. 비-upcast 블록 281–282행 `(xFF(...) .* xChunks[5]).to(of: xOut)`(fp16 곱 → fp32 덧셈).
   포트: diffusers `gate_mlp * ff_output`(fp16×fp32→fp32) / 나머지 블록 fp16×fp16 → fp32 잔차. 일치. 어텐션 쪽 게이트는 모든 블록에서 fp16 곱(247·249행) — 포트도 동일.
5. **single 블록 — 일치.** 453–456행 `FeedForward(..., upcast: false)`, `out = xUnifyheads(out) + xFF(xOut)`, `out = xIn + (out .* xChunks[2]).to(of: xIn)` — fp16 곱을 fp32 잔차에 더함, 축소 없음.
   포트: b.norm.norm `.half()` 훅만, diffusers `residual + gate*proj_out(...)`. (DT는 x_o와 FF out_proj 두 GEMM의 fp16 합, diffusers는 융합 proj_out 하나 — 수학적으로 같고 반올림만 미세 차.)
   norm_out: 685행 `normFinal(out).to(.Float16) .* scale + shift` — 포트 53·55행 일치.
6. **포트가 구현하지 않은 것(추가 축소는 없음).** FLUX.1에는 텍스트 흐름·single 블록에 별도 축소가 **없다**(×8은 위 FF와 ControlNet 주입 588·623행뿐 — schnell 비교와 무관).
   구현하지 않은 정밀도 차이는 모두 **DT가 포트보다 덜 정밀한 방향**:
   - GEMM 누산 플래그(s4nnc `GEMMFlag`, ccv `CCV_NNC_GEMM_16F` = "prefer FP16 for accumulator", `_32F` = FP32): `x_q`/`c_q` `flags: [.Float16]`(162·175행), FF `linear1` `flags: [.Float16]`(134행), FF `out_proj` `flags: [.Float32]`(140행).
     PyTorch(cuBLAS) fp16 GEMM은 fp32 누산이므로 포트는 Q 투영·FF 상향 GEMM을 DT보다 정밀하게 계산. 특히 블록 17·18의 linear1은 ×1/8 **이전**이라 fp16 누산 보호를 받지 않음. (swift-diffusion 1230c66e 예제에는 이 플래그가 없음 — 앱 코드에만 있음.)
   - 어텐션: `ScaledDotProductAttention(..., flags: [.Float16])`(229–231행, FP16 중간값; `.scale1` 모드면 q를 1/√k로 미리 곱함 — 런타임 `usesFlashAttention` 값에 달림). PyTorch SDPA는 fp32 softmax 누산.
   - q/k RMSNorm·RoPE(cmul) fp16(diffusers는 내부 fp32 후 되돌림).
   - T5 인코더의 자체 fp16 처리(swift-diffusion 363–367행 ×1/8·fp32)는 포트에서 임베딩을 bf16 파이프라인으로 미리 만든 것으로 대체 — 모든 구성에 공통 입력이라 비교 공정성에는 무관.

## swift-diffusion 1230c66e examples/flux/main.swift — 같은 논리 확인
- FeedForward 51–68행: `(1 / 8) * out` → outProjection → `.to(.Float32) * scaleFactor` (같음, 단 Dense 누산 플래그 없음).
- 249·269행 x/context `.to(.Float32)`; 155·168행 `(upcast ? xChunks[5].to(of: xOut) : xChunks[5]) .* xFF(...)` 후 `.to(of: xOut)`; 235행 single `xIn + (xChunks[2] .* out).to(of: xIn)`; 294행 `normFinal(out).to(.Float16)`.
- 차이: `(1 + scale)`을 그래프에서 계산(DT 앱은 가중치에 접어 넣음) — 정밀도 무관.
- `examples/flux_converter/main.swift`(1,012줄)는 가중치 변환용으로 **upcast·×8 처리가 없다**(515행 `(1 + scale(c)) .* normFinal(out) + shift(c)`, `.to(.Float16)` 없음). 개요가 「flux_converter에서 옮겼다」고 쓴 부분은 근거가 되지 않음 — 정밀도 처리는 examples/flux/main.swift(및 앱 Flux1.swift)에서 온다.

## 결론
1–5 모두 일치(블록 17·18 = 19개 중 마지막 둘, 조건 `i > (layers.0 - 3)` / LoRA·예제는 `i > 16`, ×8, FF 출력 fp32, 해당 블록 게이팅 fp32, single 블록 fp16 곱 → fp32 잔차, norm_out fp16).
미구현 차이(누산 fp16 플래그, fp16 어텐션 중간값, fp16 RMSNorm/RoPE)는 포트의 (b)를 **DT 실제 엔진보다 정밀·유리하게** 만드는 방향 — (b)에서 상실·오차가 나오면 실제 DT에서도 그 이상이고, (b)가 무탈해도 실제 엔진의 무탈은 보장되지 않는다는 단서가 필요하다.
원고 인용은 draw-things-community Flux1.swift(main b002b883, 파일 커밋 060f99d6; 블로그 시점 0054f779 497행)로 하는 것이 정확하다.
