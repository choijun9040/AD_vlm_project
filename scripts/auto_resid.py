"""
(R) 자동 도구 — 연산 그래프를 따라 「크기를 c배 해도 되는 영역」을 찾아 등가 교정을 구성한다. 2026-11-04.
사전 등록: thesis_outline_20260910.md §6 개정 18 ③ (결과 보기 전).

1. 기록: 작은 입력으로 모델을 한 번 실행하며 aten 연산마다 입력 텐서의 출처(앞 연산 출력·매개변수·외부)와 모듈 경로를 남긴다
   (그래프 구조는 해상도와 무관하다 — 작은 입력으로 충분하다).
2. 영역: 넘치는 텐서(측정이 찾은 연산의 출력)에서 시작해 거꾸로·앞으로 넓힌다.
   - 생산자: 선형층(addmm·mm·linear·convolution) — 입력이 영역 밖이면 「쓰는 층」(가중치·편향 ×c), 안이면 편향만 ×c.
     덧셈·뺄셈 — 텐서 입력 모두 영역(매개변수 입력은 ×c). 모양 연산(view·reshape·permute·cat·index·to 등) — 데이터 입력이 영역.
     곱셈 — 출력과 모양이 같은 피연산자만 영역(브로드캐스트되는 게이트는 그대로), 매개변수 피연산자면 그 매개변수 ×c.
     스칼라 곱·나눗셈 — 입력이 영역. 정규화 출력(아핀 있음) — 아핀 가중치·편향 ×c. 비선형 생산자 — 실패.
   - 소비자: 정규화(native_layer_norm·native_group_norm, RMSNorm 패턴 제곱→평균→+eps→rsqrt/pow(-½)→곱) — 크기에 불변이라 「읽는 쪽」, eps ×c².
     선형층 — 기본은 출구(가중치 ×1/c), 출력이 영역에 들어가야 하면 편향만 ×c로 바뀐다. 덧셈·모양·스칼라 곱·평균·합 — 출력이 영역.
     비선형 소비자 — 실패.
   - 영역이 모델 출력에 닿으면 출력에 가장 가까운 선형층·매개변수 곱을 출구로 삼아(×1/c) 다시 푼다.
3. 받아들이는 기준: fp32 등가성(출력 코사인 ≥ 0.99999). 매개변수 하나에 서로 다른 배수가 걸리면(공유 가중치 등) 실패.

    from auto_resid import trace, solve, apply_edits
    G = trace(model, lambda m: m(x_small))
    plan = solve(G, target_module="blocks.31.mlp.down_proj", target_op="addmm")
    apply_edits(model, plan, c)
"""
import math
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

LINEAR = {"addmm", "mm", "linear", "convolution", "_convolution", "conv2d", "conv1d", "cudnn_convolution", "bmm", "matmul"}
VIEWS = {"view", "_unsafe_view", "reshape", "permute", "transpose", "t", "expand", "clone", "contiguous", "_to_copy", "to", "slice",
         "select", "index", "index_select", "unsqueeze", "squeeze", "flatten", "unflatten", "split", "split_with_sizes", "chunk",
         "unbind", "cat", "stack", "alias", "detach", "narrow", "as_strided", "copy", "lift_fresh", "repeat", "pad", "constant_pad_nd",
         "upsample_nearest2d", "upsample_nearest1d", "upsample_nearest3d", "masked_fill", "where", "roll", "flip", "movedim"}
LINEAR_REDUCE = {"mean", "sum", "avg_pool2d", "avg_pool1d", "adaptive_avg_pool2d"}
NORMS = {"native_layer_norm", "native_group_norm", "layer_norm", "group_norm"}
SKIP = {"empty", "empty_like", "zeros", "ones", "full", "zeros_like", "ones_like", "full_like", "arange", "scalar_tensor", "_local_scalar_dense",
        "lift_fresh_copy", "fill_", "rand", "randn", "normal_", "uniform_", "linspace", "set_", "_to_dim_order_copy"}


class _Rec(TorchDispatchMode):
    def __init__(self, stack, pmap):
        super().__init__()
        self.stack, self.pmap = stack, pmap
        self.nodes, self.src, self.keep, self.cls = [], {}, [], {}

    def _ref(self, t):
        if id(t) in self.pmap:
            return ("param", self.pmap[id(t)])
        return self.src.get(id(t), ("ext", None))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        ins, scal = [], []

        def walk(a):
            if torch.is_tensor(a):
                ins.append(self._ref(a))
            elif isinstance(a, (list, tuple)):
                for b in a:
                    walk(b)
            elif isinstance(a, (int, float)) and not isinstance(a, bool):
                scal.append(a)
        for a in args:
            walk(a)
        for a in kwargs.values():
            walk(a)
        outs = [o for o in (out if isinstance(out, (list, tuple)) else [out]) if torch.is_tensor(o)]
        nid = len(self.nodes)
        mod = self.stack[-1] if self.stack else ""
        t0 = outs[0] if outs else None
        self.nodes.append({"op": name, "mod": mod, "cls": self.cls.get(mod, ""), "ins": ins, "scal": scal,
                           "fl": bool(t0 is not None and t0.is_floating_point() and t0.numel() > 0),
                           "shapes_in": [tuple(a.shape) for a in args if torch.is_tensor(a)],
                           "shape_out": [tuple(o.shape) for o in outs], "n_out": len(outs)})
        for k, o in enumerate(outs):
            self.src[id(o)] = ("node", (nid, k))
            self.keep.append(o)                    # id가 재사용되지 않게 붙잡아 둔다(작은 입력이라 감당된다)
        return out


def trace(model, run):
    """작은 입력으로 한 번 실행해 연산 그래프를 기록한다. run(model)이 순전파를 한다."""
    stack, hs = [], []
    for n, m in model.named_modules():
        if not n:
            continue
        hs.append(m.register_forward_pre_hook(lambda _m, _a, n=n: stack.append(n)))
        hs.append(m.register_forward_hook(lambda _m, _a, _o, n=n: (stack.pop() if stack and stack[-1] == n else None) and None))
    pmap = {id(p): n for n, p in model.named_parameters()}
    for n, b in model.named_buffers():
        pmap.setdefault(id(b), n)
    rec = _Rec(stack, pmap)
    rec.cls = {n: type(m).__name__ for n, m in model.named_modules() if n}
    try:
        with torch.no_grad(), rec:
            ret = run(model)
    finally:
        for h in hs:
            h.remove()
    nodes = rec.nodes
    cons = defaultdict(list)                       # 값 (nid, k) → [(소비 노드, 입력 위치)]
    for i, nd in enumerate(nodes):
        for j, (kind, ref) in enumerate(nd["ins"]):
            if kind == "node":
                cons[ref].append((i, j))
    # 모델 출력: run이 돌려준 텐서들(출력 객체 안의 텐서를 모두)
    found = []

    def collect(o):
        if torch.is_tensor(o):
            found.append(o)
        elif isinstance(o, (list, tuple)):
            for x in o:
                collect(x)
        elif isinstance(o, dict):
            for x in o.values():
                collect(x)
        elif hasattr(o, "__dict__"):
            for x in vars(o).values():
                collect(x)
    collect(ret)
    outputs = [rec.src[id(t)][1] for t in found if id(t) in rec.src and rec.src[id(t)][0] == "node"]
    # 매개변수에 기대는 값인가(위치로만 만든 상수 행렬과 흐름을 가르는 데 쓴다)
    dep = []
    for nd in nodes:
        dep.append(any(k == "param" or (k == "node" and dep[r[0]]) for k, r in nd["ins"]))
    rec.keep.clear()
    return {"nodes": nodes, "cons": cons, "outputs": outputs, "dep": dep}


def _param_of(G, ref, depth=6):
    """입력이 (보기 연산을 거쳐) 매개변수에서 왔으면 그 이름."""
    kind, r = ref
    while depth and kind == "node":
        nd = G["nodes"][r[0]]
        if nd["op"] not in VIEWS or not nd["ins"]:
            return None
        kind, r = nd["ins"][0]
        depth -= 1
    return r if kind == "param" else None


def _rms_reader(G, v):
    """v를 읽는 RMSNorm 패턴(제곱→평균→+eps→rsqrt/pow(-½)/sqrt→v와 곱·나눗셈)을 찾는다. (eps, 정규화된 출력 노드들, 통계 노드들, 모듈)."""
    nodes, cons = G["nodes"], G["cons"]
    for (i, _j) in cons.get(v, []):
        nd = nodes[i]
        if nd["op"] != "pow" or 2 not in nd["scal"] and 2.0 not in nd["scal"]:
            continue
        chain, cur, eps, inv = [i], (i, 0), None, None
        for _ in range(4):
            nxt = cons.get(cur, [])
            if len(nxt) != 1:
                break
            k = nxt[0][0]; chain.append(k); op = nodes[k]["op"]
            if op == "add" and nodes[k]["scal"]:
                eps = nodes[k]["scal"][0]
            if op in ("rsqrt",) or (op == "pow" and any(s < 0 for s in nodes[k]["scal"])) or op == "sqrt":
                inv = (k, 0); break
            cur = (k, 0)
        if inv is None or eps is None:
            continue
        users = [i2 for (i2, _) in cons.get(inv, []) if nodes[i2]["op"] in ("mul", "div")
                 and any(ref == ("node", v) for ref in nodes[i2]["ins"])]
        if users:
            return eps, users, chain, nodes[chain[0]]["mod"]
    return None


def targets_from_points(G, points):
    """측정 기록의 지점 이름(「모듈::연산#n」, 모듈이 비면 「<tower>」)을 그래프의 값 (노드, 0)으로 바꾼다. 못 찾은 지점은 따로 돌려준다."""
    # 측정기(headroom_ext._OpRecorder)와 똑같이 센다 — 부동소수 출력만, 건너뛰는 연산·회전 위치·정규화 내부 통계는 빼고.
    # (2026-11-04: 정수 색인 곱이 Gemma 표준화 곱 「<tower>::mul#0」으로 잘못 짝지어졌다)
    from headroom_ext import _SKIP_OPS, _NORM_STATS
    seen, idx = defaultdict(int), {}
    for i, nd in enumerate(G["nodes"]):
        mod = nd["mod"] or "<tower>"
        if (nd["op"] in _SKIP_OPS or not nd.get("fl", True) or any(k in mod for k in ("rotary", "rope"))
                or ("Norm" in nd.get("cls", "") and nd["op"] in _NORM_STATS)):
            continue
        key = f"{mod}::{nd['op']}"
        idx[f"{key}#{seen[key]}"] = i
        seen[key] += 1
    vals, miss = [], []
    for pt in points:
        if pt in idx:
            vals.append((idx[pt], 0))
        else:
            miss.append(pt)
    return vals, miss


def _setexp(scale, name, e):
    """매개변수 배수 지수를 정한다. 같은 매개변수를 여러 번 쓰면(Gemma 위치 표를 x·y로 두 번 찾아오는 등) ×c는 한 번이다 —
    누적하면 c²가 된다(2026-11-04 Gemma 등가성 코사인 0.70). ×c와 ×1/c가 겹치면 한 매개변수로 둘을 할 수 없으니 표시해 실패시킨다."""
    if scale.get(name, 0) not in (0, e):
        scale.setdefault("__conflict__", []).append(name)
    else:
        scale[name] = e


def solve(G, target_module=None, target_op=None, exits=None, max_iter=8, targets=None):
    """영역을 넓혀 교정 계획을 만든다. 반환: {"ok", "reason", "scale": {매개변수: 배수 지수(+1=×c, -1=×1/c)}, "eps": {모듈: 정규화 종류}, "region": 값 수}."""
    nodes, cons = G["nodes"], G["cons"]
    if targets:
        starts = list(targets)
    else:
        tgt = next(((i, 0) for i, nd in enumerate(nodes) if nd["mod"] == target_module and nd["op"] == target_op), None)
        if tgt is None:
            return {"ok": False, "reason": f"대상 연산을 찾지 못함: {target_module}::{target_op}"}
        starts = [tgt]
    exits = set(exits or [])
    for _it in range(max_iter):
        S, work = set(starts), list(starts)
        scale, eps, fail = defaultdict(int), {}, None
        mode = {}                                   # 선형층 노드 → "writer"/"bias"/"exit"
        inv_mul = set()                             # 정규화 패턴의 곱(출력은 불변)

        who = {}                                    # 영역 값 → 그 값을 영역에 넣은 값(실패 경로 출력용)

        def add(u):
            if u not in S:
                S.add(u); work.append(u); who[u] = v
        while work and not fail:
            v = work.pop()
            nid = v[0]
            p = nodes[nid]
            op = p["op"]
            # ── 생산자 ──
            if nid not in mode:
                if op in LINEAR:
                    w = [_param_of(G, r) for r in p["ins"]]
                    acts = [r for r, pw in zip(p["ins"], w) if pw is None and r[0] == "node"]
                    if not any(w):
                        # 위치로 만든 평균 행렬 같은 상수 × 흐름 — 매개변수에 기대는 쪽이 흐름이다(Gemma 풀링)
                        flow = [r for r in acts if G["dep"][r[1][0]]]
                        if len(flow) != 1:
                            fail = f"가중치 없는 행렬곱에서 흐름을 정할 수 없다: {p['mod']}::{op}"; break
                        add(flow[0][1]); mode[nid] = "linear_map"
                    elif any(r[1] in S for r in acts):
                        mode[nid] = "bias"
                    else:
                        mode[nid] = "writer"
                elif op in ("add", "sub"):
                    if len(p["ins"]) < 2 and any(s != 0 for s in p["scal"][:1]):
                        fail = f"상수를 더하는 연산이 영역에 있다: {p['mod']}::{op}"; break
                    for r in p["ins"]:
                        if r[0] == "node":
                            add(r[1])
                        elif r[0] == "param":
                            _setexp(scale, r[1], 1)
                        else:
                            fail = f"영역 덧셈에 외부 값이 들어온다: {p['mod']}::{op}"
                elif op in ("mul", "div"):
                    tens = [(r, s) for r, s in zip(p["ins"], p["shapes_in"])]
                    params = [r for r, _ in tens if r[0] == "param" or (_param_of(G, r) is not None)]
                    if len(tens) == 1:
                        if tens[0][0][0] == "node":
                            add(tens[0][0][1])
                    elif params and op == "mul":
                        others = [r for r, _ in tens if r not in params and r[0] == "node"]
                        if any(r[1] in S for r in others):
                            pass                     # 흐름 × 매개변수(layer scale 류) — 흐름이 이미 줄어 있다
                        else:
                            pr = params[0]; _setexp(scale, pr[1] if pr[0] == "param" else _param_of(G, pr), 1)
                    else:
                        full = [r for r, s in tens if s == p["shape_out"][0]]
                        if op == "div":
                            full = [tens[0][0]]
                        if len(full) != 1:
                            fail = f"곱의 어느 쪽을 줄일지 정할 수 없다: {p['mod']}::{op}"; break
                        if full[0][0] == "node":
                            add(full[0][1])
                elif op in VIEWS or op in LINEAR_REDUCE:
                    if p["ins"] and p["ins"][0][0] == "node":
                        if op in ("cat", "stack"):
                            for r in p["ins"]:
                                if r[0] == "node":
                                    add(r[1])
                        elif op == "where":
                            for r in p["ins"][1:]:
                                if r[0] == "node":
                                    add(r[1])
                        else:
                            add(p["ins"][0][1])
                    elif p["ins"] and p["ins"][0][0] == "param":
                        _setexp(scale, p["ins"][0][1], 1)
                    else:
                        fail = f"영역이 모델 입력에 닿는다: {p['mod']}::{op}"
                elif op == "embedding":
                    # 표 찾아오기 — 출력은 표(매개변수) 값 그대로라 표를 c배 하면 출력도 c배다(위치 임베딩 등). 쓰는 층으로 다룬다
                    tbl = _param_of(G, p["ins"][0]) if p["ins"] else None
                    if tbl is None:
                        fail = f"매개변수가 아닌 표를 찾아온다: {p['mod']}::{op}"; break
                    _setexp(scale, tbl, 1)
                    mode[nid] = "embed_writer"
                elif op in ("zeros", "zeros_like", "empty", "empty_like") or (op in ("scalar_tensor", "full", "full_like", "fill_")
                                                                             and p["scal"] and all(x == 0 for x in p["scal"][:1])):
                    # 값이 0인 상수 — c배 해도 0이라 영역에 들어와도 바꿀 것이 없다(where(패딩, 0, x) 등)
                    mode[nid] = "zero_const"
                elif op in NORMS:
                    ws = [r[1] for r in p["ins"][1:] if r[0] == "param"]
                    if not ws:
                        fail = f"아핀 없는 정규화 출력이 영역에 있다: {p['mod']}::{op}"; break
                    for w in ws:
                        _setexp(scale, w, 1)
                    mode[nid] = "norm_writer"
                else:
                    fail = f"비선형 연산이 영역을 만든다: {p['mod']}::{op}"; break
            # ── 소비자 ──
            rr = _rms_reader(G, v)
            if rr:
                eps[rr[3]] = "rms"
                inv_mul.update(rr[1])
                skip = set(rr[1]) | set(rr[2])
            else:
                skip = set()
            for (q, j) in cons.get(v, []):
                if q in skip:
                    continue
                c = nodes[q]; cop = c["op"]
                if cop in NORMS and j == 0:
                    eps[c["mod"]] = cop
                    continue
                if cop in LINEAR and not any(_param_of(G, r) for r in c["ins"]):
                    for k in range(c["n_out"]):
                        add((q, k))
                    continue
                if cop in LINEAR:
                    if q in exits:
                        mode[q] = "exit"; continue
                    if mode.get(q) is None:
                        mode[q] = "exit_pending"
                    continue
                if cop in ("add", "sub", "cat", "stack", "where") or cop in VIEWS or cop in LINEAR_REDUCE:
                    for k in range(c["n_out"]):
                        add((q, k))
                    continue
                if cop in ("mul", "div"):
                    if q in exits:
                        mode[q] = "exit_mul"; continue
                    if c["shape_out"] and c["shapes_in"] and c["shapes_in"][j] != c["shape_out"][0] and len(c["shapes_in"]) > 1:
                        fail = f"영역 값이 브로드캐스트되는 쪽으로 곱해진다: {c['mod']}::{cop}"; break
                    for k in range(c["n_out"]):
                        add((q, k))
                    continue
                if cop in SKIP:
                    continue
                pv = nodes[v[0]]
                path, u = [], v
                while u is not None and len(path) < 25:
                    path.append(f"{nodes[u[0]]['mod']}::{nodes[u[0]]['op']}#{u[0]}"); u = who.get(u)
                fail = f"비선형 연산이 영역 값을 직접 읽는다: {c['mod']}::{cop} (읽는 값: {pv['mod']}::{pv['op']} #{v[0]}, 입력 위치 {j}; 경로 {' ← '.join(path)})"; break
        if fail:
            return {"ok": False, "reason": fail, "region": len(S)}
        # 생산자 선형층의 쓰는 층/편향만 판정은 영역이 다 찬 뒤에 다시 한다(처리 순서에 따라 입력이 늦게 영역에 들어올 수 있다)
        for q, m in list(mode.items()):
            if m in ("writer", "bias"):
                acts = [r for r in nodes[q]["ins"] if r[0] == "node" and _param_of(G, r) is None]
                mode[q] = "bias" if any(r[1] in S for r in acts) else "writer"
        # 출구로 남은 선형층: 출력이 영역에 들어갔으면 편향만, 아니면 가중치 ×1/c
        for q, m in list(mode.items()):
            if m == "exit_pending":
                mode[q] = "bias" if any((q, k) in S for k in range(nodes[q]["n_out"])) else "exit"
        # 모델 출력이 영역에 닿았는가 → 출력에 가장 가까운 선형층·매개변수 곱을 출구로
        hit = [o for o in G["outputs"] if o in S]
        if hit:
            cand = None
            for o in hit:
                cur = o
                for _ in range(64):
                    nd = nodes[cur[0]]
                    if nd["op"] in LINEAR or (nd["op"] == "mul" and any(r[0] == "param" or _param_of(G, r) for r in nd["ins"])):
                        cand = cur[0]; break
                    nxt = [r[1] for r in nd["ins"] if r[0] == "node" and r[1] in S]
                    if not nxt:
                        break
                    cur = nxt[0]
            if cand is None or cand in exits:
                return {"ok": False, "reason": "영역이 모델 출력에 닿는데 출구로 삼을 층이 없다", "region": len(S)}
            exits.add(cand)
            continue
        # 배수 모으기
        for q, m in mode.items():
            nd = nodes[q]
            ws = [(_param_of(G, r), r) for r in nd["ins"]]
            pw = [w for w, _ in ws if w]
            if nd["op"] in LINEAR:
                weight = [w for w in pw if not w.endswith("bias")]
                bias = [w for w in pw if w.endswith("bias")]
                if m == "writer":
                    for w in weight + bias:
                        _setexp(scale, w, 1)
                elif m == "bias":
                    for w in bias:
                        _setexp(scale, w, 1)
                elif m == "exit":
                    for w in weight:
                        _setexp(scale, w, -1)
            elif m == "exit_mul":
                for w in pw:
                    _setexp(scale, w, -1)
        if scale.get("__conflict__"):
            return {"ok": False, "reason": f"한 매개변수가 ×c와 ×1/c를 함께 요구한다: {sorted(set(scale['__conflict__']))[:5]}", "region": len(S)}
        return {"ok": True, "scale": {k: v for k, v in scale.items() if v}, "eps": eps, "region": len(S),
                "exits": sorted(exits), "writers": sorted({nodes[q]["mod"] for q, m in mode.items() if m == "writer"}),
                "bias_only": sorted({nodes[q]["mod"] for q, m in mode.items() if m == "bias"}),
                "exit_layers": sorted({nodes[q]["mod"] for q, m in mode.items() if m in ("exit", "exit_mul")})}
    return {"ok": False, "reason": "출구 찾기가 수렴하지 않았다"}


@torch.no_grad()
def apply_edits(model, plan, c):
    """계획대로 매개변수를 c^지수배, 정규화 eps를 c²배 한다. 바꾼 eps 목록을 돌려준다(가중치 파일 밖 설정)."""
    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))
    for name, e in plan["scale"].items():
        params[name].mul_(c ** e)
    mods = dict(model.named_modules())
    changed = {}
    for m in plan["eps"]:
        mod = mods[m]
        for attr in ("eps", "variance_epsilon", "epsilon"):
            if isinstance(getattr(mod, attr, None), float):
                old = getattr(mod, attr)
                setattr(mod, attr, old * c * c)
                changed[m] = {"attr": attr, "old": old, "new": old * c * c}
                break
        else:
            raise RuntimeError(f"eps 속성을 찾지 못함: {m}")
    return changed


def choose_c(mx, fmax=65504.0, target=2.0):
    return min(1.0, 2.0 ** math.floor(math.log2(fmax / (target * mx)))) if mx > 0 else 1.0
