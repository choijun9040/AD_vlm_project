"""
AWQ 로딩 호환 shim
====================
transformers 4.49.0에는 `transformers.models.qwen3`가 없는데, autoawq 0.2.9의
`awq/models/__init__.py`가 이를 무조건 import 한다. 그 결과 `import awq` 자체가
ModuleNotFoundError로 죽고, transformers의 AWQ 통합(`integrations/awq.py`)이
`from awq.modules.linear.gemm import WQLinear_GEMM`을 하지 못해 INT4 체크포인트를
아예 로드할 수 없다.

INT4 추론에 필요한 건 `WQLinear_GEMM` 하나뿐이므로, awq 패키지가 참조만 하고
실제로는 쓰지 않는 qwen3 계열 클래스를 빈 스텁으로 sys.modules에 미리 꽂아
import 사슬만 통과시킨다. 양자화(quantize)에는 쓰지 말 것 — 추론 전용 우회다.

사용법: 모델을 로드하기 전에 `import awq_compat; awq_compat.patch()` 한 번.
"""

import sys
import types

_STUBS = {
    "qwen3": ("modeling_qwen3", ["Qwen3DecoderLayer", "Qwen3ForCausalLM"]),
    "qwen3_moe": (
        "modeling_qwen3_moe",
        ["Qwen3MoeDecoderLayer", "Qwen3MoeForCausalLM", "Qwen3MoeSparseMoeBlock"],
    ),
}

_patched = False


def patch() -> bool:
    """스텁이 필요하면 꽂고 True, 이미 정상이거나 적용됐으면 False를 반환."""
    global _patched
    if _patched:
        return False

    import transformers.models as tm

    applied = False
    for pkg_name, (mod_name, class_names) in _STUBS.items():
        full_pkg = f"transformers.models.{pkg_name}"
        try:
            __import__(f"{full_pkg}.{mod_name}")
            continue  # 진짜 모듈이 있으면 건드리지 않는다
        except ImportError:
            pass

        pkg = types.ModuleType(full_pkg)
        mod = types.ModuleType(f"{full_pkg}.{mod_name}")
        for cls in class_names:
            setattr(mod, cls, type(cls, (), {}))
        setattr(pkg, mod_name, mod)
        sys.modules[full_pkg] = pkg
        sys.modules[f"{full_pkg}.{mod_name}"] = mod
        setattr(tm, pkg_name, pkg)
        applied = True

    _patched = True
    if applied:
        print("[awq_compat] transformers에 qwen3 모듈이 없어 스텁을 적용했다 (INT4 추론 전용)")
    return applied


def use_reference_kernel() -> None:
    """AWQ 행렬곱을 순수 PyTorch 참조 구현으로 강제한다.

    AWQ의 forward는 awq_ext(CUDA) → triton → naive 순으로 커널을 고른다.
    이 환경에는 awq_ext가 없고, torch 2.3에 딸려온 triton에는 AWQ 커널이 쓰는
    `tl.interleave`가 없어 런타임에 AttributeError로 죽는다. TRITON_AVAILABLE을
    꺼서 `dequantize_gemm`(정확한 역양자화 + fp16 matmul) 경로로 보낸다 —
    느리지만 INT4 가중치의 정의 그대로라 진단용으로는 오히려 이쪽이 맞다.
    """
    from awq.modules.linear import gemm

    gemm.TRITON_AVAILABLE = False
    print("[awq_compat] AWQ 커널을 순수 PyTorch 참조 구현으로 고정했다 (느림, 수치는 정확)")
