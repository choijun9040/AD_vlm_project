"""
headroom_guard 확장 — 연산 단위 진단과 잔차 재조정(등가 교정). 2026-10-30 신설.
사전 등록: thesis_outline_20260910.md §6 「교정의 전 계열 확장」(2026-10-30 23:20 KST, 결과 보기 전).

**왜 필요한가.**
(a) 기존 진단은 블록 출력과 타워 최종 출력만 본다. Gemma 4처럼 블록 뒤의 **함수형 연산**(pooler의 `× √d`, 표준화의 뺄셈)에서
    넘치는 계열은 모듈 hook에 잡히지 않고, 타워 최종 출력은 `× std_scale` 뒤라 작아 **안전으로 오판한다.**
    → 디스패치 단계에서 모든 연산의 출력 크기를 기록해 **처음 형식 상한을 넘는 연산**을 찾는다.
(b) 기존 교정은 게이트형 MLP의 곱셈 이전 축소(비등가)뿐이다.
    → 잔차에 값을 쓰는 곳 전부를 c배 하고 출구에서 되돌리는 **잔차 재조정**(등가)을 더한다. 잔차를 읽는 곳이
    스케일 불변 정규화(RMSNorm·LayerNorm)이면 중간 계산은 그대로이고, 잔차 흐름과 그로부터 선형으로 만들어지는 값만 c배가 된다.

**적용 여부는 구조 추측이 아니라 등가성 검사로 정한다** — 이름 규칙으로 후보를 만들고, fp32에서 교정 전후 최종 출력을
비교해(코사인·크기 비율) 통과한 후보만 쓴다. 통과하는 후보가 없으면 「잔차 재조정 불가」로 보고한다.
"""
import copy
import math
import statistics as st

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

# 값이 바뀌지 않는 연산(보기 바꾸기·복사)은 기록하지 않는다 — 가중치 전치 같은 것이 활성처럼 섞이지 않게
_SKIP_OPS = {"t", "view", "_unsafe_view", "expand", "permute", "transpose", "reshape", "detach",
             "alias", "slice", "select", "unsqueeze", "squeeze", "as_strided", "clone", "contiguous",
             "_to_copy", "copy_", "lift_fresh", "split", "split_with_sizes", "unbind", "chunk",
             "index_select", "cat", "stack", "flatten", "unflatten", "new_empty", "empty_like",
             "zeros_like", "ones_like", "full_like", "arange", "lt", "gt", "le", "ge", "eq", "ne"}


_NORM_STATS = {"pow", "mean", "rsqrt", "var", "var_mean", "sum", "add", "sqrt"}


class _OpRecorder(TorchDispatchMode):
    """연산마다 (지점 이름, max|x|, 승격 여부)를 기록한다.

    - 정규화 모듈 안의 분산 통계(x², 평균, rsqrt)는 활성이 아니라 내부 통계이고 참조 구현이 fp32로 계산하므로 뺀다
      (2026-10-30 시험에서 RMSNorm의 x²가 「처음 초과」로 잡혔다).
    - 모델 형식(bf16)보다 넓은 형식(fp32)으로 계산된 연산은 **승격 연산**으로 따로 표시한다. 참조 구현이 승격하는
      곳(Gemma 4의 pooler·표준화)은 그 승격이 없는 배포 경로에서만 위험하므로, 판정을 두 범주로 나눈다.
    """
    def __init__(self, stack, model_dtype):
        super().__init__()
        self.stack, self.seq, self.count, self.model_dtype = stack, [], {}, model_dtype

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name in _SKIP_OPS:
            return out
        t = out[0] if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]) else out
        if torch.is_tensor(t) and t.is_floating_point() and t.numel() > 0:
            mod, cls = self.stack[-1] if self.stack else ("<tower>", "")
            if any(k in mod for k in ("rotary", "rope")):
                return out               # 회전 위치 부호화는 이미지 내용과 무관한 위치 계산이다 — 활성이 아니다
            if "Norm" in cls and name in _NORM_STATS:
                return out
            base = f"{mod}::{name}"
            n = self.count.get(base, 0)
            self.count[base] = n + 1
            # 어텐션 마스크는 의도적으로 -inf·finfo.min(≈3.4e38)을 채운다 — 크기는 bf16에서 재므로 진짜 활성이
            # 무한대가 될 일은 없다. 절댓값 1e30 이상은 마스크로 보고 뺀다(2026-10-30 시험에서 마스크가 초과로 잡혔다)
            a = t.detach().abs().float()
            up = t.dtype != self.model_dtype and t.element_size() > torch.tensor([], dtype=self.model_dtype).element_size()
            self.seq.append((f"{base}#{n}", torch.where(a < 1e30, a, torch.zeros_like(a)).amax(), up))
        return out


def _install_stack(root, stack):
    """모듈 경로 스택. **hook은 반드시 None을 돌려준다** — forward hook이 값을 돌려주면 PyTorch가
    그 값으로 모듈 출력을 바꿔 끼운다(2026-10-30 시험에서 pop()의 반환값이 출력이 되어 터졌다)."""
    hs = []
    for name, m in root.named_modules():
        if not name:
            continue

        def pre(_m, _a, n=name, c=type(m).__name__):
            stack.append((n, c))

        def post(_m, _a, _o, n=name):
            if stack and stack[-1][0] == n:
                stack.pop()
        hs.append(m.register_forward_pre_hook(pre))
        hs.append(m.register_forward_hook(post))
    return hs


@torch.no_grad()
def profile_ops(tower, cache, fwd, fmax, target=2.0):
    """모든 연산 출력의 이미지별 max|x|. 실행 순서대로 지점 목록과 처음 넘는 지점을 돌려준다.

    크기는 넘치지 않는 형식(bf16)에서 재고 `fmax`(목표 형식의 상한 또는 가상 상한)로 판정한다.
    """
    stack, order, vals, upcast = [], [], {}, {}
    mdt = next(p for p in tower.parameters() if p.is_floating_point()).dtype
    hs = _install_stack(tower, stack)
    try:
        for x in cache:
            rec = _OpRecorder(stack, mdt)
            with rec:
                fwd(tower, x)
            stack.clear()
            got = torch.stack([v for _, v, _ in rec.seq]).cpu().tolist() if rec.seq else []
            for (k, _, up), v in zip(rec.seq, got):
                if k not in vals:
                    vals[k] = []
                    order.append(k)
                    upcast[k] = up
                vals[k].append(v)
    finally:
        for h in hs:
            h.remove()
    n = len(cache)
    rows = []
    for k in order:
        v = vals[k]
        fin = [x for x in v if math.isfinite(x)]
        worst = max(fin) if fin else float("inf")
        exceed = (sum(1 for x in fin if x > fmax) + (len(v) - len(fin))) / n
        rows.append({"point": k, "upcast": upcast[k],
                     "max_p50": st.median(fin) if fin else float("inf"), "max_worst": worst,
                     "headroom_worst": fmax / worst if worst and math.isfinite(worst) else 0.0,
                     "exceed_rate": exceed})
    # 모델 형식 연산에서 처음 넘는 곳(그대로 배포하면 넘친다)과, 승격 연산까지 포함해 처음 넘는 곳
    # (참조 구현의 fp32 승격이 없는 경로에서 넘친다)을 나눠 돌려준다
    first = next((r for r in rows if r["exceed_rate"] > 0 and not r["upcast"]), None)
    first_any = next((r for r in rows if r["exceed_rate"] > 0), None)
    top = max(rows, key=lambda r: r["max_worst"]) if rows else None
    risky = [r for r in rows if r["headroom_worst"] < target]
    return {"n_points": len(rows), "first_exceed": first, "first_exceed_no_upcast": first_any,
            "global_max": top,
            "risky_points": risky, "fmax": fmax, "rows": rows}


# ---------------------------------------------------------------- 잔차 재조정
ATTN_CHILD = ("attn", "self_attn", "attention")
ATTN_OUT = ("proj", "o_proj", "out_proj", "wo", "dense")
MLP_CHILD = ("mlp", "feed_forward", "ffn")
MLP_OUT = ("down_proj", "fc2", "w2")
SANDWICH = ("post_attention_layernorm", "post_feedforward_layernorm")      # Gemma: x + post_norm(branch)
LAYERSCALE = (("ls1", "ls2"), ("layer_scale1", "layer_scale2"), ("lambda_1", "lambda_2"))  # InternViT 등
PROJECTOR = ("multi_modal_projector", "mlp1", "img_projection", "mm_projector", "visual_projection",
             "connector", "projector")


def _child(m, names):
    for n in names:
        c = getattr(m, n, None)
        if c is not None:
            return n, c
    return None, None


def _affine(mod):
    """모듈이 가진 가중치·편향(있는 것만)."""
    out = []
    for pn in ("weight", "bias"):
        p = getattr(mod, pn, None)
        if isinstance(p, torch.nn.Parameter) or torch.is_tensor(p):
            out.append(p)
    return out


def block_writers(block):
    """블록에서 잔차에 값을 쓰는 텐서들과 그 방식. 못 찾으면 None."""
    if all(hasattr(block, n) for n in SANDWICH):
        ts = []
        for n in SANDWICH:
            ts += _affine(getattr(block, n))
        return "sandwich_norm", ts
    for a, b in LAYERSCALE:
        pa, pb = getattr(block, a, None), getattr(block, b, None)
        if torch.is_tensor(pa) and torch.is_tensor(pb):
            return "layer_scale", [pa, pb]
    _, attn = _child(block, ATTN_CHILD)
    _, mlp = _child(block, MLP_CHILD)
    if attn is None or mlp is None:
        return None
    _, ao = _child(attn, ATTN_OUT)
    _, mo = _child(mlp, MLP_OUT)
    if mo is None and hasattr(mlp, "fc0") and hasattr(mlp, "fc1"):
        mo = mlp.fc1                     # MoonViT(Kimi-VL)의 MLP는 fc0 → fc1 — 출력층이 fc1이다
    if ao is None or mo is None:
        return None
    return "out_proj", _affine(ao) + _affine(mo)


class _Stop(Exception):
    pass


@torch.no_grad()
def find_entry_candidates(tower, blocks, x, fwd):
    """block0 입력을 만든 모듈 후보 — 실행 순서상 block0 앞이고 출력의 값 분포가 block0 입력과 같은 것.

    Qwen2.5-VL은 패치 임베딩 뒤 창 순서로 행을 재배열하므로 메모리 주소나 위치별 값이 아니라
    **합·제곱합·최대**(재배열 불변량)로 맞춘다.
    """
    b0 = blocks[0]
    seen, cap = [], {}

    def inv(t):
        t = t.detach().double()
        return (t.sum().item(), t.pow(2).sum().item(), t.abs().max().item(), t.numel())

    def pre_b0(_m, args, kwargs=None):
        t = args[0] if args else next(v for v in (kwargs or {}).values() if torch.is_tensor(v))
        cap["b0"] = inv(t)
        raise _Stop()                # block0에 들어가는 순간 멈춘다

    hs = []
    for name, m in tower.named_modules():
        if not name or m is b0 or name.startswith(_block_prefix(tower, blocks)):
            continue

        def post(_m, _a, o, n=name, mm=m):
            t = o[0] if isinstance(o, (tuple, list)) and o and torch.is_tensor(o[0]) else o
            if torch.is_tensor(t) and t.is_floating_point():
                seen.append((n, mm, inv(t)))
        hs.append(m.register_forward_hook(post))
    hs.append(b0.register_forward_pre_hook(pre_b0, with_kwargs=True))
    try:
        fwd(tower, x)
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    if "b0" not in cap:
        return []
    s, q, mx, n = cap["b0"]

    def close(a, b):
        return abs(a - b) <= 1e-3 * max(1.0, abs(b))
    idx = [i for i, (nm, mm, (s2, q2, m2, n2)) in enumerate(seen)
           if n2 == n and close(s2, s) and close(q2, q) and close(m2, mx)]
    cands = []
    for i in reversed(idx):                       # 가장 마지막에 끝난(가장 바깥) 일치 모듈부터
        cands.append([seen[i][0]])
    # 입구 모듈이 바깥에서 받은 값에 매개변수를 더하는 경우(GLM-4.1V의 embeddings가 위치 임베딩을 더한다) —
    # 그 값을 만든 바로 앞 모듈(출력 크기가 같은 것)까지 함께 c배 하는 후보
    for i in reversed(idx):
        for j in range(i - 1, -1, -1):
            if seen[j][2][3] == n and not seen[i][0].startswith(seen[j][0] + "."):
                cands.append([seen[j][0], seen[i][0]])
                break
    return cands


def _block_prefix(tower, blocks):
    for name, m in tower.named_modules():
        if m is blocks:
            return name + "."
    return "\0"


def entry_tensors(mod):
    """입구 모듈에서 c배 할 텐서. 정규화·선형·합성곱·임베딩이면 그 가중치·편향, 합성 모듈이면
    안의 선형·합성곱·임베딩 잎과 위치·클래스 임베딩 매개변수."""
    cls = type(mod).__name__
    if "Norm" in cls or isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Embedding)):
        return _affine(mod)
    ts = []
    for n, sub in mod.named_modules():
        if isinstance(sub, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Embedding)):
            ts += _affine(sub)
    for n, p in mod.named_parameters(recurse=False):
        if any(k in n.lower() for k in ("pos", "position", "class", "cls")):
            ts.append(p)
    return ts


def exit_kind(tower):
    if hasattr(tower, "std_bias") and hasattr(tower, "std_scale"):
        return "standardize"
    return None


EPS_ATTRS = ("variance_epsilon", "eps", "epsilon")


def _eps_attr(mod):
    for a in EPS_ATTRS:
        v = getattr(mod, a, None)
        if isinstance(v, float):
            return a
    return None


@torch.no_grad()
def find_residual_norms(tower, tower_c, x, fwd, c, tol=0.02):
    """잔차를 읽는 정규화 — 가중치만 c배 한 모델에서 **입력 크기가 c배가 된** 정규화 모듈 이름.

    **왜 필요한가 (2026-10-31, 사전 등록 개정).** 정규화는 `x / √(mean(x²) + eps)`라 입력이 c배가 되면 mean(x²)는 c²배인데
    eps는 그대로여서, 활성이 작은 앞쪽 블록에서 불변이 미세하게 깨진다(Qwen2.5-VL 실측: 코사인 0.99979). 이 정규화들의 eps를
    c²배 하면 정확한 재매개화가 된다(같은 4장에서 코사인 1.0). 가지 안의 정규화(Gemma 사후 정규화, q·k 정규화)는 입력이
    바뀌지 않으므로 eps를 건드리면 안 된다 — 그래서 구조로 추측하지 않고 입력 크기 비율을 실측해 고른다.
    """
    def collect(t):
        rec = {}
        hs = []
        for n, m in t.named_modules():
            if _eps_attr(m) is None:
                continue

            def h(_m, args, n=n):
                a = args[0] if args else None
                if torch.is_tensor(a):
                    rec[n] = rec.get(n, 0.0) + float(a.detach().double().pow(2).sum())
            hs.append(m.register_forward_pre_hook(h))
        fwd(t, x)
        for h in hs:
            h.remove()
        return rec
    r0, r1 = collect(tower), collect(tower_c)
    out = []
    for n, v0 in r0.items():
        if v0 > 0 and n in r1:
            ratio = math.sqrt(r1[n] / v0)
            if abs(ratio / c - 1) < tol:
                out.append(n)
    return out


@torch.no_grad()
def apply_eps(tower, norm_names, c):
    """잔차를 읽는 정규화의 eps를 c²배 (모듈 설정이라 state_dict에 실리지 않는다 — 배포 쪽에서 함께 적용할 것)."""
    mods = dict(tower.named_modules())
    changed = {}
    for n in norm_names:
        a = _eps_attr(mods[n])
        old = getattr(mods[n], a)
        setattr(mods[n], a, old * c * c)
        changed[n] = {"attr": a, "old": old, "new": old * c * c}
    return changed


def choose_c(max_value, fmax, target=2.0):
    """목표 여유를 넘기는 가장 큰 2의 거듭제곱 — 곱해도 가수가 그대로라 반올림이 생기지 않는다."""
    if not (max_value and math.isfinite(max_value)):
        return None
    return min(1.0, 2.0 ** math.floor(math.log2(fmax / (target * max_value))))


@torch.no_grad()
def apply_residual(tower, blocks_of, entry_names, c, inplace=False):
    """잔차 재조정을 적용한다(기본은 사본). entry_names는 c배 할 입구 모듈 이름 목록,
    blocks_of(tower)는 블록 리스트를 돌려주는 함수."""
    t = tower if inplace else copy.deepcopy(tower)
    mods = dict(t.named_modules())
    for en in entry_names:
        for p in entry_tensors(mods[en]):
            p.mul_(c)
    for blk in blocks_of(t):
        kind_ts = block_writers(blk)
        if kind_ts is None:
            raise RuntimeError("잔차 쓰기 층을 찾지 못한 블록이 있다")
        for p in kind_ts[1]:
            p.mul_(c)
    if exit_kind(t) == "standardize":
        t.std_bias.mul_(c)
        t.std_scale.div_(c)
    return t


@torch.no_grad()
def compare(t0, t1, cache, fwd, comp=1.0):
    """최종 출력의 이미지별 코사인과 크기 비율(교정 쪽에 comp를 곱해 비교)."""
    cos, ratio = [], []
    for x in cache:
        a = fwd(t0, x).double().flatten()
        b = (fwd(t1, x).double() * comp).flatten()
        if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
            cos.append(float("nan")); ratio.append(float("nan")); continue
        cos.append(float(F.cosine_similarity(a, b, dim=0)))
        ratio.append(float(b.norm() / a.norm()))
    fin = [v for v in cos if math.isfinite(v)]
    rf = [v for v in ratio if math.isfinite(v)]
    return {"cos_min": min(fin) if fin else float("nan"),
            "ratio_min": min(rf) if rf else float("nan"),
            "ratio_max": max(rf) if rf else float("nan"),
            "n_nonfinite": len(cos) - len(fin)}


def plan_residual(tower32, blocks_of, probe_cache32, fwd, c, cos_tol=0.99999):
    """입구 후보마다 잔차 재조정을 시험해, 등가가 되는 첫 후보를 고른다.

    출구 처리 세 가지 — (i) 타워 안 최종 정규화가 흡수(크기 비율 1), (ii) 표준화 보정(Gemma),
    (iii) 정규화 없이 프로젝터로 넘김(크기 비율 c) → 프로젝터 첫 선형층에 1/c(여기서는 comp=1/c로 비교).
    """
    blocks = blocks_of(tower32)
    kinds = [block_writers(b) for b in blocks]
    if any(k is None for k in kinds):
        bad = [i for i, k in enumerate(kinds) if k is None]
        return {"ok": False, "reason": f"잔차 쓰기 층을 찾지 못한 블록 {bad[:5]}"}
    cands = find_entry_candidates(tower32, blocks, probe_cache32[0], fwd)
    if not cands:
        return {"ok": False, "reason": "block0 입력을 만든 모듈을 찾지 못함"}
    tried = []
    for names in cands:
        try:
            t1 = apply_residual(tower32, blocks_of, names, c)
        except RuntimeError as e:
            tried.append({"entry": names, "error": str(e)}); continue
        norms = find_residual_norms(tower32, t1, probe_cache32[0], fwd, c)
        apply_eps(t1, norms, c)
        r = compare(tower32, t1, probe_cache32, fwd)
        exit_ = None
        if abs(r["ratio_min"] - 1) < 1e-3 and abs(r["ratio_max"] - 1) < 1e-3:
            exit_ = "standardize" if exit_kind(tower32) == "standardize" else "final_norm"
            comp = 1.0
        elif abs(r["ratio_min"] / c - 1) < 1e-3 and abs(r["ratio_max"] / c - 1) < 1e-3:
            exit_, comp = "projector", 1.0 / c
        tried.append({"entry": names, **r, "exit": exit_, "n_eps_norms": len(norms)})
        if exit_ and r["cos_min"] >= cos_tol:
            return {"ok": True, "entry": names, "exit": exit_, "comp": comp, "eps_norms": norms,
                    "writers": kinds[0][0], "tried": tried}
        del t1
    return {"ok": False, "reason": "등가가 되는 입구 후보가 없음", "tried": tried}


def projector_first(model):
    """부모 모델의 프로젝터에서 첫 매개 층(선형 또는 정규화)."""
    if model is None:
        return None, None
    for n in PROJECTOR:
        p = getattr(model, n, None) or getattr(getattr(model, "model", None), n, None)
        if p is None:
            continue
        for sn, sub in p.named_modules():
            if isinstance(sub, torch.nn.Linear) or "Norm" in type(sub).__name__:
                return f"{n}.{sn}" if sn else n, sub
    return None, None


@torch.no_grad()
def compensate_projector(model, c):
    """출구가 프로젝터이면 첫 층이 정규화일 때는 그대로, 선형일 때는 가중치를 1/c배(편향은 그대로)."""
    name, sub = projector_first(model)
    if sub is None:
        return {"done": False, "reason": "프로젝터를 찾지 못함 — 배포 쪽에서 첫 선형층 가중치를 1/c배 할 것"}
    if "Norm" in type(sub).__name__:
        return {"done": True, "layer": name, "how": "정규화가 흡수"}
    sub.weight.div_(c)
    return {"done": True, "layer": name, "how": f"가중치 ×{1/c:g}"}


def virtual_fmax(ops_profile, ratio=1.5):
    """가상 상한 — 그 모델의 최악 최대 ÷ ratio. 실제로 넘지 않는 계열에서 교정의 효과를 보기 위한 판정 기준."""
    g = ops_profile["global_max"]
    return g["max_worst"] / ratio if g else None


def summarize_ops(p, k=8):
    """저장용 요약 — 전체 행은 크므로 처음 넘는 지점, 최대 지점, 위험 지점 상위만."""
    risky = sorted(p["risky_points"], key=lambda r: -r["max_worst"])[:k]
    return {"n_points": p["n_points"], "fmax": p["fmax"], "first_exceed": p["first_exceed"],
            "first_exceed_no_upcast": p["first_exceed_no_upcast"],
            "global_max": p["global_max"], "n_risky": len(p["risky_points"]), "risky_top": risky}
