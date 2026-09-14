"""
Qwen2.5-VL 비전 타워 ONNX 내보내기 (Orin Nano 검증용)
========================================================
검증하려는 명제는 "fp16 연산 경로에서 비전 타워 마지막 블록이 표현 범위를 넘는가"이므로
**LLM은 필요 없다.** 비전 타워만 내보내 TensorRT fp16 엔진으로 만들고, A100에서 쓴 것과
같은 이미지로 붕괴율을 재면 된다. 절차 전체는 `ORIN_DEPLOY_PLAN.md` 참조.

**핵심 어려움과 해법**
비전 타워의 `forward(hidden_states, grid_thw)`는 `grid_thw`로부터 회전 위치 임베딩,
윈도우 인덱스, `cu_seqlens`를 매번 계산한다. 여기에 파이썬 리스트·`unique_consecutive`·
`argsort`·파이썬 루프가 섞여 있어 그대로는 ONNX로 잘 나가지 않는다.

배포 해상도는 어차피 고정하므로, **grid에 의존하는 모든 값을 미리 계산해 상수로 굳히고**
블록 연산만 남긴 래퍼를 내보낸다. attention이 `cu_seqlens`로 매번 만드는 블록 대각
마스크도 미리 만들어 버퍼로 넣는다(그래서 attention forward를 하나 우회한다).

내보낸 그래프는 원본 타워와 수치가 일치하는지 **반드시 대조**한 뒤 저장한다.

실행:
    python scripts/export_vision_tower_onnx.py \
        --checkpoint checkpoints/student_baseline_v2/epoch_1 \
        --max_pixels 200704 --out onnx/tower_baseline_v2.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

sys.path.append(str(Path(__file__).parent))
import awq_compat
from dataloader import build_token_to_images

BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
DRIVELM_VAL_JSON = "data/QA_dataset_nus/v1_1_val_nus_q_only.json"


# --------------------------------------------------------------------------
# attention 우회 — cu_seqlens로 매번 마스크를 만드는 대신 미리 만든 것을 쓴다
# --------------------------------------------------------------------------
def _attn_forward_with_const_mask(self, hidden_states, cu_seqlens=None,
                                  rotary_pos_emb=None, position_embeddings=None):
    """`cu_seqlens` 자리로 **미리 만든 가산 마스크**를 받는다(None이면 마스크 없음).

    모듈 속성(`self._export_mask`)으로 넘기면 트레이서가 블록마다 별개의 상수로
    구워 버려, 원본 해상도에서 203 MiB짜리 마스크가 28벌 복제된다(실측 5.83 GiB).
    블록의 forward가 `cu_seqlens`를 attn까지 그대로 전달하므로, 이 통로로 넘기면
    **그래프 값 하나를 28개 블록이 공유**한다."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
    import math

    seq_length = hidden_states.shape[0]
    q, k, v = (self.qkv(hidden_states)
               .reshape(seq_length, 3, self.num_heads, -1)
               .permute(1, 0, 2, 3).unbind(0))
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

    mask = cu_seqlens                   # (1, S, S) 또는 None(전 구간 한 덩어리)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    w = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
    if mask is not None:
        w = w + mask
    w = nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(w, v).transpose(0, 1).reshape(seq_length, -1)
    return self.proj(out)


class TowerExportWrapper(nn.Module):
    """grid 의존 값을 전부 상수로 굳힌 비전 타워. 입력은 pixel_values 하나뿐이다."""

    def __init__(self, tower, grid_thw):
        super().__init__()
        self.tower = tower
        dev = next(tower.parameters()).device

        rot = tower.rot_pos_emb(grid_thw)
        window_index, cu_window = tower.get_window_index(grid_thw)
        cu_window = torch.unique_consecutive(
            torch.tensor(cu_window, device=dev, dtype=torch.int32))
        cu_full = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_full = nn.functional.pad(cu_full, (1, 0), value=0)

        seq_len = int(grid_thw[:, 0].mul(grid_thw[:, 1]).mul(grid_thw[:, 2]).sum())
        emb = torch.cat((rot.reshape(seq_len // tower.spatial_merge_unit,
                                     tower.spatial_merge_unit, -1)[window_index, :, :]
                         .reshape(seq_len, -1),) * 2, dim=-1)

        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.register_buffer("window_index", window_index.to(dev), persistent=False)
        self.register_buffer("reverse_index", torch.argsort(window_index).to(dev),
                             persistent=False)
        self.merge_unit = tower.spatial_merge_unit
        self.seq_len = seq_len

        # 블록별 마스크를 미리 만들어 attention에 심는다
        # 마스크 채움값을 -1e4로 둔다. 원본은 finfo(float32).min(-3.4e38)을 쓰는데
        # 그 값은 **fp16에서 -inf가 되어** 우리가 찾으려는 것과 무관한 NaN을 만든다.
        # exp(-1e4)는 fp32에서도 이미 0이라 softmax 결과는 동일하다(대조로 확인한다).
        # 원본 해상도(seq_len≈7,300)에서는 마스크 하나가 200 MiB를 넘는다. 블록마다
        # 따로 들면 6 GiB가 넘으므로 **구간 id 벡터(수십 KB)만 버퍼로 두고** 마스크는
        # forward에서 한 번 만들어 32개 블록이 같은 텐서를 참조하게 한다. 상수 접기는
        # 그래프 값 하나를 접으므로 초기화값도 종류당 하나만 남는다.
        def seg_ids(cu):
            s = torch.zeros(seq_len, device=dev, dtype=torch.int32)
            for i in range(1, len(cu)):
                s[int(cu[i - 1]):int(cu[i])] = i - 1
            return s

        # 구간이 하나뿐이면(=이미지 한 장의 full attention) 마스크가 전부 0이라 불필요하다
        self.seg_full = None if len(cu_full) <= 2 else seg_ids(cu_full)
        self.seg_win = None if len(cu_window) <= 2 else seg_ids(cu_window)
        for name, t in (("seg_full", self.seg_full), ("seg_win", self.seg_win)):
            if t is not None:
                self.register_buffer("_" + name, t, persistent=False)
        self.fullatt_idx = set(tower.fullatt_block_indexes)
        for blk in tower.blocks:
            blk.attn.forward = _attn_forward_with_const_mask.__get__(blk.attn)

    @staticmethod
    def _mask_from_seg(seg, fill=-1e4):
        if seg is None:
            return None
        eq = seg.unsqueeze(0).unsqueeze(-1) == seg.unsqueeze(0).unsqueeze(0)
        return torch.where(eq, 0.0, fill).to(torch.float32)

    def forward(self, pixel_values):
        t = self.tower
        # 마스크 두 종류를 한 번만 만들어 모든 블록이 같은 그래프 값을 쓰게 한다
        m_full = self._mask_from_seg(self.seg_full)
        m_win = self._mask_from_seg(self.seg_win)
        h = t.patch_embed(pixel_values)
        h = h.reshape(self.seq_len // self.merge_unit, self.merge_unit, -1)
        h = h[self.window_index, :, :].reshape(self.seq_len, -1)
        pos = (self.cos, self.sin)
        for i, blk in enumerate(t.blocks):
            h = blk(h, cu_seqlens=(m_full if i in self.fullatt_idx else m_win),
                    position_embeddings=pos)
        h = t.merger(h)
        return h[self.reverse_index, :]


def dedup_constants(model, min_bytes=2**20):
    """내용이 같은 큰 `Constant` 노드를 **초기화값 하나**로 합친다.

    트레이서는 윈도우 마스크처럼 입력과 무관한 큰 텐서를 소비 블록마다 별개의
    `Constant` 노드로 굽는다. 원본 해상도에서는 203 MiB × 28벌 = 5.83 GiB가 되어
    파일이 2.5 GiB → 8.3 GiB로 부푼다. `do_constant_folding=False`로도,
    마스크를 인자로 넘겨도 막지 못했으므로(둘 다 실측) **결과 그래프를 직접 고친다**.

    `onnx.save_model(convert_attribute=True)`가 노드 attribute까지 외부 데이터로
    쓰기 때문에, 초기화값 개수만 세면 이 부풀기는 보이지 않는다.
    """
    import hashlib
    import onnx as _onnx

    seen, rename, drop = {}, {}, []
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        t = next((a.t for a in node.attribute
                  if a.name == "value" and len(a.t.raw_data) >= min_bytes), None)
        if t is None:
            continue
        # shape·dtype까지 키에 넣는다 — 바이트가 같아도 형상이 다르면 합치면 안 된다
        digest = (tuple(t.dims), t.data_type, hashlib.sha1(t.raw_data).hexdigest())
        if digest not in seen:
            name = f"_shared_const_{len(seen)}"
            init = _onnx.TensorProto()
            init.CopyFrom(t)
            init.name = name
            model.graph.initializer.append(init)
            seen[digest] = name
        rename[node.output[0]] = seen[digest]
        drop.append(node)

    if not drop:
        return 0
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rename:
                node.input[i] = rename[inp]
    for node in drop:
        model.graph.node.remove(node)
    return len(drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/student_baseline_v2/epoch_1",
                    help="LoRA 어댑터 경로. 'none'이면 베이스 모델 그대로")
    ap.add_argument("--max_pixels", type=int, default=200704,
                    help="배포 해상도. 이 값으로 형상이 고정된다")
    ap.add_argument("--min_pixels", type=int, default=50176)
    ap.add_argument("--out", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--tol", type=float, default=2e-3, help="원본 대비 허용 최대 오차")
    ap.add_argument("--fold", choices=["auto", "on", "off"], default="auto",
                    help="상수 접기. auto=마스크가 64 MiB를 넘으면 끔")
    ap.add_argument("--fixa", type=float, default=0.0, metavar="SCALE",
                    help="마지막 블록에 fixA를 적용해 내보낸다. gate·up 가중치를 각각 "
                         "1/sqrt(SCALE)로 나눠 곱을 1/SCALE로 만든다(MLP에 bias가 없어 "
                         "순전파 훅과 등가). 0이면 적용하지 않는다. 논문 7.4용.")
    args = ap.parse_args()

    awq_compat.patch()
    out = Path(args.out or f"onnx/tower_{Path(args.checkpoint).parent.name}.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[로드] {BASE} (+ {args.checkpoint}) — eager attention, fp32/CPU→CUDA")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32, attn_implementation="eager")
    if args.checkpoint.lower() != "none":
        model = PeftModel.from_pretrained(model, args.checkpoint).merge_and_unload()
    tower = model.visual.eval().to("cuda")

    if args.fixa:
        import math as _math
        mlp = tower.blocks[-1].mlp
        f = 1.0 / _math.sqrt(args.fixa)
        # 주의: Qwen2.5-VL **비전** MLP는 `Qwen2_5_VLMLP(config, bias=True)`로 만들어져
        # bias가 있다(LLM 쪽과 다르다). 원래의 fixA는 gate/up의 **출력**을 스케일하는
        # 순전파 훅이므로, 등가인 가중치 편집은 weight와 bias를 **함께** 나누는 것이다.
        # weight만 나누면 bias가 남아 훅과 다른 그래프가 된다.
        with torch.no_grad():
            for lin in (mlp.gate_proj, mlp.up_proj):
                lin.weight.mul_(f)
                if lin.bias is not None:
                    lin.bias.mul_(f)
        has_b = mlp.gate_proj.bias is not None
        print(f"[fixA] 마지막 블록 gate·up의 weight{'·bias' if has_b else ''}를 각각 "
              f"x{f:.6f} (곱 1/{args.fixa:g}) — 논문 7.4 처방을 그래프에 굳혔다")

    # 대표 이미지 하나로 형상을 고정한다
    tok = build_token_to_images(DRIVELM_VAL_JSON)
    img_path = str(next(iter(tok.values()))["CAM_FRONT"])
    proc = AutoProcessor.from_pretrained(BASE, max_pixels=args.max_pixels,
                                         min_pixels=args.min_pixels)
    enc = proc.image_processor(images=[Image.open(img_path).convert("RGB")],
                               return_tensors="pt")
    pv = enc["pixel_values"].to("cuda", torch.float32)
    grid = enc["image_grid_thw"].to("cuda")
    tokens = int(grid.prod(dim=1).sum()) // 4
    print(f"  고정 형상: pixel_values={tuple(pv.shape)}  grid={grid.tolist()}  "
          f"vision token ≈ {tokens}")

    with torch.no_grad():
        ref = tower(pv, grid)
    wrapper = TowerExportWrapper(tower, grid).eval()
    with torch.no_grad():
        got = wrapper(pv)

    diff = (ref.float() - got.float()).abs().max().item()
    print(f"[대조] 원본 대비 최대 오차 = {diff:.3e}  (허용 {args.tol:.1e})")
    if diff > args.tol:
        raise SystemExit("래퍼가 원본과 다르다 — 내보내기 중단. 상수화 로직을 확인할 것")
    print("  일치 확인 — 상수화가 원본 동작을 보존한다")

    # 상수 접기를 켜면 윈도우 마스크가 **소비 블록마다 Constant 노드 하나씩** 복제된다.
    # 원본 해상도(S=7,296)에서 마스크 하나가 203 MiB이고 윈도우 블록이 28개라
    # 파일이 2.5 GiB → 8.3 GiB로 부푼다(실측). 마스크가 클 때는 접기를 끄고
    # 구간 id 벡터(수십 KB)에서 런타임에 만들게 둔다 — TensorRT가 알아서 처리한다.
    seq_len = int(grid.prod(dim=1).sum())
    mask_mib = seq_len * seq_len * 4 / 2**20
    fold = {"on": True, "off": False}.get(args.fold, mask_mib <= 64)
    if not fold:
        print(f"[내보내기] 상수 접기 끔 — 마스크 {mask_mib:.0f} MiB가 블록마다 복제되는 것을 막는다")

    print(f"[내보내기] {out}")
    import tempfile, shutil, onnx
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.onnx"
        torch.onnx.export(
            wrapper, (pv,), str(raw), opset_version=args.opset,
            input_names=["pixel_values"], output_names=["image_embeds"],
            do_constant_folding=fold,
        )
        m = onnx.load(str(raw))

    def _big_consts(g, min_bytes=2**20):
        for n in g.node:
            if n.op_type != "Constant":
                continue
            for a in n.attribute:
                if a.name == "value" and len(a.t.raw_data) >= min_bytes:
                    yield n, a.t

    before = sum(len(t.raw_data) for _, t in _big_consts(m.graph))
    dropped = dedup_constants(m)
    after = sum(len(t.raw_data) for _, t in _big_consts(m.graph))
    if dropped:
        print(f"  중복 Constant {dropped}개를 공유 초기화값으로 합쳤다: "
              f"{before/2**20:.0f} MiB → {after/2**20:.0f} MiB")
    elif before:
        print(f"  큰 Constant 노드 합계 {before/2**20:.0f} MiB (중복 없음)")

    # 기본 내보내기는 가중치를 수백 개 파일로 흩뿌린다. Orin으로 옮기기 쉽게 한 파일로 모은다.
    for f in out.parent.glob(out.stem + ".onnx.data"):
        f.unlink()
    onnx.save_model(m, str(out), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=out.name + ".data",
                    size_threshold=1024, convert_attribute=True)
    data = out.parent / (out.name + ".data")
    print(f"  완료: {out.name} ({out.stat().st_size/2**20:.1f} MiB) "
          f"+ {data.name} ({data.stat().st_size/2**30:.2f} GiB)")

    # ONNX 그래프가 PyTorch와 같은 값을 내는지 CPU에서 확인.
    # 주의: 절대오차로 판정하면 안 된다. 블록 32개를 지나며 누적되는 fp32 재정렬 오차만으로도
    # 최대 절대오차가 0.8 수준까지 벌어지며, 이는 **같은 PyTorch 코드를 CPU와 CUDA에서
    # 돌려도 동일하게 나타난다**(실측 0.8394). 따라서 방향 일치도(코사인)로 판정하고,
    # CPU/CUDA 차이를 기준선으로 함께 찍는다.
    try:
        import onnxruntime as ort
        ref_cpu = got.float().cpu()
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        o = torch.from_numpy(sess.run(None, {"pixel_values": pv.cpu().numpy()})[0])
        cos = torch.nn.functional.cosine_similarity(o, ref_cpu, dim=-1).median().item()
        amax = (o - ref_cpu).abs().max().item()
        print(f"[대조] ONNX 런타임: 토큰별 코사인 중앙값 = {cos:.6f}  "
              f"(참고: 최대 절대오차 {amax:.3f})")
        if cos < 0.9999:
            print("  경고: 방향이 어긋난다 — TensorRT 빌드 전에 원인을 확인할 것")
        else:
            print("  일치 확인 — 절대오차는 fp32 누적 차이 수준이다")
    except ImportError:
        print("  (onnxruntime 없음 — 그래프 검증 생략)")

    print(f"""
다음 단계 (Orin에서):
  trtexec --onnx={out.name} --saveEngine=tower_fp16.plan --fp16
  trtexec --onnx={out.name} --saveEngine=tower_fp32.plan            # 대조군
  trtexec --onnx={out.name} --saveEngine=tower_bf16.plan --bf16     # 지원 시

  같은 250장을 넣어 출력에 NaN/Inf가 나오는 이미지 비율을 센다.
  A100 기준값: student_baseline_v2 94.8% / student_full 0.0% (원본 해상도).
  이 스크립트는 --max_pixels로 해상도를 고정하므로 비교하려는 해상도와 맞출 것.

검증 도구가 없으면 (선택):
  pip install onnx onnxruntime   # ONNX 그래프 자체 점검용
""")


if __name__ == "__main__":
    main()
