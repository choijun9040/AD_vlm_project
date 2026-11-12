"""Qwen2.5-VL 비전 어텐션(SDPA) 마스크를 층마다 다시 만들지 않게 한다 — 수치는 그대로, 속도만. 2026-11-10.

transformers 4.49 `Qwen2_5_VLVisionSdpaAttention.forward`는 층마다 [1, L, L] bool 마스크를 새로 만들고, 창마다 `cu_seqlens[i]`
(GPU 텐서)로 자르느라 동기화를 수천 번 한다. 같은 `cu_seqlens`(창 단위·전체 두 종류)와 같은 L이면 마스크 내용이 같으므로
한 번 만들어 재사용한다. SDPA에 들어가는 q·k·v·마스크가 원래와 같아 출력이 비트 단위로 같아야 한다(적용 전 검사로 확인).

    import qwen_vision_mask_cache; qwen_vision_mask_cache.patch()
"""
import torch
import torch.nn.functional as F

_CACHE = {}


def _mask(cu_seqlens, L, device):
    b = cu_seqlens.tolist()                                      # 층마다 동기화 한 번(원래는 창마다 두 번)
    key = (tuple(b), L, str(device))                             # 내용으로 찾는다 — 주소 재사용에 속지 않게
    m = _CACHE.get(key)
    if m is None:
        _CACHE.clear() if len(_CACHE) > 8 else None
        m = torch.zeros([1, L, L], device=device, dtype=torch.bool)
        for i in range(1, len(b)):
            m[..., b[i - 1]:b[i], b[i - 1]:b[i]] = True
        _CACHE[key] = m
    return m


def patch():
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as Q
    if getattr(Q.Qwen2_5_VLVisionSdpaAttention, "_mask_cache_patched", False):
        return
    apply_rope = Q.apply_rotary_pos_emb_vision

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
        L = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(L, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos().float(), emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)
        attention_mask = _mask(cu_seqlens, L, q.device)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        o = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        return self.proj(o.transpose(0, 1).reshape(L, -1))

    Q.Qwen2_5_VLVisionSdpaAttention.forward = forward
    Q.Qwen2_5_VLVisionSdpaAttention._mask_cache_patched = True
