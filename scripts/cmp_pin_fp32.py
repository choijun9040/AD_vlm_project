"""
E-B — TensorRT 「Accuracy Considerations」·Polygraphy 「Work with Reduced Precision」의 절차대로 문제 층을 찾아 fp32로 고정한다. 2026-11-04.
사전 등록: thesis_outline_20260910.md §6 개정 15 (결과 보기 전).

절차(되풀이):
  1. polygraphy run <p2> --trt --strongly-typed --trt-outputs mark all --validate  — 모든 중간 텐서 가운데 inf/NaN이 생긴 것을 찾는다
  2. 그래프 순서(ONNX 노드는 위상 정렬)에서 가장 앞의 텐서를 만든 노드를 fp32로 고정한다 — fp16 입력은 Cast(FLOAT), fp16 가중치는 fp32 사본,
     출력은 fp32로 두고 아직 fp16인 소비자 앞에는 Cast(FLOAT16)을 넣는다. 고정할 노드가 그 Cast라면 Cast를 지우고 소비자들을 고정한다
  3. inf/NaN이 없어질 때까지 1–2를 되풀이한다(강타입 fp16 그래프에 Cast로 고정 — 관행 빌드는 원고에서 뺐다)
넘침은 블록 31(분할 그래프의 뒤쪽 p2)에서 생긴다. p2 입력은 polygraphy로 p1을 돌려 저장한다(--save-outputs → --load-inputs).
venv_pg(polygraphy, venv_trt 패키지를 읽기만)에서 돈다.

    /workspace/venv_pg/bin/python scripts/cmp_pin_fp32.py --p1 onnx/cmp/tower_unfixed_nativeh_p1.onnx --p2 onnx/cmp/tower_unfixed_nativeh_p2.onnx \
        --npz onnx/cmp/probe8.npz --work onnx/cmp/pin --out eval_results/cmp/eb_pin.json
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PG = "/workspace/venv_pg/bin/polygraphy"
LOADER = '''import numpy as np
z = np.load("{npz}")
def load_data():
    for k in sorted(k for k in z.files if not k.startswith("_")):
        yield {{"pixel_values": z[k]}}
'''


def sh(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, Path(log).read_text()


def bad_tensors(text):
    """polygraphy --validate 출력에서 NaN/Inf가 난 텐서 이름을 모은다."""
    names = set()
    for line in text.splitlines():
        m = re.search(r"Output:\s*'([^']+)'.*?(?:Inf|NaN|inf|nan)", line) or re.search(r"\[(.+?)\].*(?:Encountered (?:one or more )?(?:Inf|NaN))", line)
        if m:
            names.add(m.group(1))
        m2 = re.search(r"Inf|NaN", line)
        if m2 and "| " in line:
            names.add(line.split("|")[0].strip().split()[-1])
    return names


class Pinner:
    def __init__(self, path):
        self.m = onnx.load(path, load_external_data=True)
        self.g = self.m.graph
        self.inits = {i.name: i for i in self.g.initializer}
        self.f32_alias = {}        # fp16 텐서 이름 → 같은 값의 fp32 텐서 이름(고정한 노드의 출력)
        self.cast_of = {}          # 우리가 넣은 Cast(FLOAT16) 출력 → 그 입력(fp32)
        self.pinned = []
        self.n = 0

    def _uid(self, base):
        self.n += 1
        return f"{base}__pin{self.n}"

    def producers(self):
        return {o: nd for nd in self.g.node for o in nd.output}

    def order(self):
        return {o: i for i, nd in enumerate(self.g.node) for o in nd.output}

    def pin_node(self, nd):
        if nd.op_type == "Cast" and nd.output[0] in self.cast_of:
            # 우리가 넣은 fp16 되돌리기 Cast가 넘친다 — 그 소비자들을 fp32로 고정한다
            out = nd.output[0]
            cons = [c for c in self.g.node if out in c.input]
            for c in cons:
                self.pin_node(c)
            return
        new_inputs = []
        pre = []
        for x in nd.input:
            if x == "":
                new_inputs.append(x); continue
            if x in self.f32_alias:
                new_inputs.append(self.f32_alias[x]); continue
            if x in self.inits and self.inits[x].data_type == TensorProto.FLOAT16:
                nm = self._uid(x + "_f32")
                arr = numpy_helper.to_array(self.inits[x]).astype(np.float32)
                self.g.initializer.append(numpy_helper.from_array(arr, nm)); self.inits[nm] = self.g.initializer[-1]
                new_inputs.append(nm); continue
            # 활성 입력 — 형식을 알 수 없으면 FLOAT로 Cast(이미 fp32면 TensorRT가 그대로 둔다)
            nm = self._uid(x + "_f32")
            pre.append(helper.make_node("Cast", [x], [nm], to=TensorProto.FLOAT, name=self._uid("pin_cast_in")))
            new_inputs.append(nm)
        del nd.input[:]; nd.input.extend(new_inputs)
        # 노드 속성 가운데 형식을 정하는 것(Cast to)은 FLOAT로
        for at in nd.attribute:
            if nd.op_type == "Cast" and at.name == "to" and at.i == TensorProto.FLOAT16:
                at.i = TensorProto.FLOAT
        post = []
        for k, y in enumerate(list(nd.output)):
            y32 = self._uid(y + "_f32")
            nd.output[k] = y32
            self.f32_alias[y] = y32
            post.append(helper.make_node("Cast", [y32], [y], to=TensorProto.FLOAT16, name=self._uid("pin_cast_out")))
            self.cast_of[y] = y32
        idx = list(self.g.node).index(nd)
        for j, p in enumerate(pre):
            self.g.node.insert(idx + j, p)
        idx = list(self.g.node).index(nd)
        for j, p in enumerate(post):
            self.g.node.insert(idx + 1 + j, p)
        # 형식 정보가 남아 있으면 TensorRT 파서가 어긋난다 — 바꾼 텐서의 value_info를 지운다
        keep = [v for v in self.g.value_info if v.name not in set(nd.output) | set(self.f32_alias)]
        del self.g.value_info[:]; self.g.value_info.extend(keep)
        self.pinned.append({"node": nd.name, "op": nd.op_type})

    def save(self, path):
        onnx.save(self.m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=Path(path).name + ".data", size_threshold=1024)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--p1", required=True)
    ap.add_argument("--p2", required=True)
    ap.add_argument("--npz", required=True, help="탐색용 pixel_values(prep_trt_inputs.py)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_iter", type=int, default=12)
    a = ap.parse_args()
    W = Path(a.work); W.mkdir(parents=True, exist_ok=True)
    (W / "loader.py").write_text(LOADER.format(npz=str(Path(a.npz).resolve())))
    res = {"p1": a.p1, "p2": a.p2, "npz": a.npz, "iters": []}

    # p1을 돌려 p2 입력을 저장한다(p1 쪽에 inf가 없는지도 본다)
    p1o = W / "p1_out.json"
    if not p1o.exists():
        rc, txt = sh([PG, "run", a.p1, "--trt", "--strongly-typed", "--data-loader-script", str(W / "loader.py"),
                      "--validate", "--save-outputs", str(p1o)], W / "p1.log")
        res["p1_validate_rc"] = rc
        print(f"[p1] 종료 {rc}", flush=True)

    cur = a.p2
    pin = None
    for it in range(a.max_iter):
        rc, txt = sh([PG, "run", cur, "--trt", "--strongly-typed", "--load-inputs", str(p1o), "--trt-outputs", "mark", "all",
                      "--validate"], W / f"iter{it}.log")
        bad = bad_tensors(txt)
        passed = rc == 0 and not bad
        print(f"[되풀이 {it}] 종료 {rc} · inf/NaN 텐서 {len(bad)}", flush=True)
        if pin is None:
            pin = Pinner(a.p2)
        order = pin.order()
        cand = sorted((order[t], t) for t in bad if t in order)
        res["iters"].append({"iter": it, "rc": rc, "n_bad": len(bad), "first": cand[0][1] if cand else None})
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        if passed:
            break
        if not cand:
            res["stopped"] = "inf/NaN 텐서를 그래프에서 찾지 못했다 — 로그 확인"
            break
        nd = pin.producers()[cand[0][1]]
        pin.pin_node(nd)
        cur = str(W / f"p2_pin{it + 1}.onnx")
        pin.save(cur)
    res["pinned"] = pin.pinned if pin else []
    res["final_p2"] = cur
    # 표시 없이(엔진 그대로) 마지막 그래프를 한 번 더 검증한다
    rc, txt = sh([PG, "run", cur, "--trt", "--strongly-typed", "--load-inputs", str(p1o), "--validate"], W / "final.log")
    res["final_validate_rc"] = rc
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "iters"}, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
