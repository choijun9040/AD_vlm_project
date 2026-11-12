"""
Draw Things 원형 처방(DT-원형)을 PyTorch 비전 타워에 훅으로 재현 — 개정 14b (2026-11-04, thesis_outline §6, 결과 보기 전 등록).

근거: Draw Things 공개 모델 코드(liuliu/swift-diffusion `examples/flux/main.swift`, 2026-11-04 확인)
  - 잔차 흐름 fp32: `out = xEmbedder(x).to(.Float32)`, 블록 안은 `norm(x).to(.Float16)`로 fp16 계산, `x + (gate .* out).to(of: x)`
  - 다운 프로젝션: `out = (1 / 8) * out` → `outProjection(out)` → `out.to(.Float32) * 8` (편향은 변환기가 그대로 복사 → 결과적으로 편향 ×s)
  - 블록: `upcast: i > 16` — 「Just last layer should be enough, but to be safe, for the last 2 layers」
게이팅은 Qwen·Gemma 비전 블록에 없다(게이트 = 1).

    from dt_original import apply_dt_qwen
    hooks = apply_dt_qwen(tower16, s=8)      # tower16: fp16 Qwen2.5-VL 비전 타워
    ...
    for h in hooks: h.remove()
"""
import torch


def _to_half(_m, _i, o):
    return o.half() if torch.is_tensor(o) else o


def _to_float(_m, _i, o):
    return o.float() if torch.is_tensor(o) else o


def _down_hooks(lin, s):
    """다운 프로젝션 입력 ×1/s(fp16) — 출력은 fp32로 올려 ×s. 편향은 건드리지 않는다(원형 그대로)."""
    pre = lin.register_forward_pre_hook(lambda _m, a: (a[0] * (1.0 / s),) + tuple(a[1:]))
    post = lin.register_forward_hook(lambda _m, _i, o: o.float() * float(s))
    return [pre, post]


def apply_dt_qwen(tower, s=8, last_n=2):
    """Qwen2.5-VL 비전 타워(fp16)에 DT-원형을 건다. 돌려준 훅을 지우면 원래대로."""
    hooks = [tower.patch_embed.register_forward_hook(_to_float)]          # 잔차 흐름 fp32
    blocks = tower.blocks
    for i, b in enumerate(blocks):
        hooks.append(b.norm1.register_forward_hook(_to_half))             # 블록 안은 fp16
        hooks.append(b.norm2.register_forward_hook(_to_half))
        if i >= len(blocks) - last_n:
            hooks += _down_hooks(b.mlp.down_proj, s)
    hooks.append(tower.merger.ln_q.register_forward_hook(_to_half))       # 출구 정규화 뒤 fp16
    return hooks


def apply_dt_gemma(vm, s=8, last_n=2):
    """Gemma 4 비전 인코더(fp16)에 DT-원형을 건다 — 잔차 fp32, 층 입력 정규화 출력은 fp16, 마지막 2층 다운 프로젝션 입력 ×1/s·출력 fp32 ×s."""
    hooks = [vm.patch_embedder.register_forward_hook(_to_float)]
    # 회전 위치 임베딩은 잔차 형식을 따라 만들어진다 — 블록 안 계산(fp16)에 맞춰 fp16으로(원형도 어텐션을 Float16으로 한다)
    hooks.append(vm.encoder.rotary_emb.register_forward_hook(
        lambda _m, _i, o: tuple(t.half() for t in o) if isinstance(o, tuple) else o.half()))
    layers = vm.encoder.layers
    for i, lyr in enumerate(layers):
        for nm in ("input_layernorm", "pre_feedforward_layernorm"):
            hooks.append(getattr(lyr, nm).register_forward_hook(_to_half))
        # 사후 정규화는 fp16 분기 출력을 받아 잔차에 더한다 — 출력을 fp32로(잔차와 같은 형식)
        for nm in ("post_attention_layernorm", "post_feedforward_layernorm"):
            hooks.append(getattr(lyr, nm).register_forward_hook(_to_float))
        if i >= len(layers) - last_n:
            dp = lyr.mlp.down_proj
            lin = dp if isinstance(dp, torch.nn.Linear) else dp.linear      # Gemma4ClippableLinear 감싸개
            hooks += _down_hooks(lin, s)
            if lin is not dp:
                # 감싸개의 출력 자르기는 fp16 범위 기준이다 — fp32 출력에서 그대로 두면 원형과 다르다. 감싸개 출력도 fp32로 받는다
                hooks.append(dp.register_forward_hook(_to_float))
    return hooks


# ── TensorRT 「Accuracy Considerations」의 두 처방 (개정 15c, 2026-11-04) ─────────────────────────────────────────────
class _FP32Linear(torch.nn.Module):
    """넘치는 계산을 fp32로 — 입력을 fp32로 올리고 fp32 가중치로 계산해 fp32로 내보낸다(문서: 강타입에서 ICastLayer로 FP32 계산)."""
    def __init__(self, lin32):
        super().__init__()
        # 매개변수·버퍼로 등록하지 않는다 — 바깥에서 .half()를 불러도 fp32로 남게(내보내기 스크립트가 일치 검사 중 형식을 바꾼다)
        self.w32 = lin32.weight.detach().float().clone()
        self.b32 = None if lin32.bias is None else lin32.bias.detach().float().clone()

    def forward(self, x):
        return torch.nn.functional.linear(x.float(), self.w32.to(x.device), None if self.b32 is None else self.b32.to(x.device))


def apply_trt_mixed_qwen(tower16, tower32, blocks=(31,)):
    """B1 — 블록 `blocks`의 down_proj를 fp32로 계산하고, 그 결과가 정규화(merger ln_q)를 만나기 전까지의 잔차 덧셈을 fp32로 둔다(fp32 섬).
    tower32: 같은 모델의 fp32 타워(가중치 원본). 돌려준 되돌리기 함수를 부르면 원래대로."""
    saved = []
    for bi in blocks:
        mlp = tower16.blocks[bi].mlp
        saved.append((mlp, mlp.down_proj))
        mlp.down_proj = _FP32Linear(tower32.blocks[bi].mlp.down_proj).to(next(tower16.parameters()).device)
    h = tower16.merger.ln_q.register_forward_hook(_to_half)       # 정규화 뒤부터 다시 fp16

    def undo():
        h.remove()
        for mlp, dp in saved:
            mlp.down_proj = dp
    return undo


def apply_input_scale_qwen(tower16, s=0.25):
    """B2 — 문서의 「Scale the input data」: 입력 pixel_values만 s배(다른 곳은 그대로)."""
    h = tower16.patch_embed.register_forward_pre_hook(lambda _m, a: (a[0] * s,) + tuple(a[1:]))
    return h.remove
