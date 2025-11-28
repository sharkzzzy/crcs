"""
core/attention_processor.py
FINAL V10: Cross-Attention Gating (Region Control).
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ... ContextBank, AlphaScheduler, LayerSelector, _infer_scale_from_layer_id 保持不变 ...
# 请直接保留之前的定义
class ContextBank:
    def __init__(self):
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}
    def clear(self):
        self._store.clear()
    def put(self, layer_id: str, K: torch.Tensor, V: torch.Tensor):
        self._store[layer_id] = {"K": K.detach().to(dtype=torch.float16), "V": V.detach().to(dtype=torch.float16)}
    def has(self, layer_id: str) -> bool:
        return layer_id in self._store
    def get(self, layer_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._store[layer_id]["K"], self._store[layer_id]["V"]

class AlphaScheduler:
    def __init__(self, construct_phase=(0.0, 0.4, 0.2), texture_phase=(0.4, 0.8, 0.8), refine_phase=(0.8, 1.0, 0.5), per_layer_scaling=None):
        self.phases = [construct_phase, texture_phase, refine_phase]
        self.per_layer_scaling = per_layer_scaling or {"lowres": 1.0, "midres": 0.6, "hires": 0.35}
    def alpha(self, step_idx: int, total_steps: int, layer_id: str) -> float:
        frac = step_idx / max(total_steps - 1, 1)
        val = self.phases[-1][2]
        is_refine = False
        for start, end, a in self.phases:
            if start <= frac <= end:
                val = a
                if start >= 0.8: is_refine = True
                break
        lid = layer_id or ""
        if "down_blocks.2" in lid or "mid_block" in lid or "up_blocks.0" in lid:
            scale = self.per_layer_scaling.get("midres", 0.6)
        elif "down_blocks.0" in lid or "down_blocks.1" in lid:
            scale = self.per_layer_scaling.get("lowres", 1.0)
        else:
            scale = self.per_layer_scaling.get("hires", 0.35) if is_refine else 0.0
        return float(val * scale)

class LayerSelector:
    def __init__(self, patterns=None, probe_patterns=None):
        self.patterns = patterns or ["down_blocks.2", "mid_block", "up_blocks.0", "up_blocks.2"]
        self.probe_patterns = probe_patterns or ["down_blocks.2", "mid_block"]
    def for_inject(self, layer_id: str) -> bool:
        return layer_id and any(p in layer_id for p in self.patterns)
    def for_probe(self, layer_id: str) -> bool:
        return layer_id and any(p in layer_id for p in self.probe_patterns)

def _infer_scale_from_layer_id(layer_id: str) -> int:
    if not layer_id: return 1
    if "mid_block" in layer_id: return 8
    m = re.search(r"down_blocks.(\d+)", layer_id)
    if m: return 2 ** int(m.group(1))
    m2 = re.search(r"up_blocks.(\d+)", layer_id)
    if m2: return {0: 4, 1: 2, 2: 1}.get(int(m2.group(1)), 1)
    return 1

# ==============================================================================
# MAIN PROCESSOR (With Gating)
# ==============================================================================
class CARCAttentionProcessor(nn.Module):
    def __init__(
        self,
        context_bank: Optional[ContextBank] = None,
        alpha_scheduler: Optional[AlphaScheduler] = None,
        layer_selector: Optional[LayerSelector] = None,
    ):
        super().__init__()
        self.context_bank = context_bank or ContextBank()
        self.alpha_scheduler = alpha_scheduler or AlphaScheduler()
        self.layer_selector = layer_selector or LayerSelector()

        self.mode: str = "off"
        self.probe_enabled: bool = False
        self.step_idx: int = 0
        self.total_steps: int = 50
        self.subject_id: Optional[int] = None
        self.base_latent_hw: Optional[Tuple[int, int]] = None
        
        self.subject_token_ids: Dict[int, List[int]] = {}
        self._attn_accumulator: Dict[int, torch.Tensor] = {}
        
        # 【新增】区域门控掩码
        self._region_mask: Optional[torch.Tensor] = None # [1,1,H0,W0]

    # --- Setters ---
    def set_mode(self, mode: str): self.mode = mode
    def set_probe_enabled(self, flag: bool):
        self.probe_enabled = bool(flag)
        if self.probe_enabled: self._attn_accumulator.clear()
    def set_step_index(self, idx: int, total: int):
        self.step_idx, self.total_steps = int(idx), int(total)
    def set_subject_id(self, i: Optional[int]): self.subject_id = i
    def set_subject_token_ids(self, mapping: Dict[int, List[int]]): self.subject_token_ids = mapping or {}
    def set_base_latent_hw(self, h: int, w: int): self.base_latent_hw = (int(h), int(w))
    
    # 【新增】设置区域掩码
    def set_region_mask(self, mask: Optional[torch.Tensor]):
        self._region_mask = mask

    @torch.no_grad()
    def pop_attn_maps(self) -> Dict[int, torch.Tensor]:
        out = {}
        for sid, amap in self._attn_accumulator.items():
            mi, ma = amap.min(), amap.max()
            out[sid] = (amap - mi) / (ma - mi + 1e-8)
        self._attn_accumulator.clear()
        return out

    # --- Helpers ---
    def _align_bg_batch(self, key, K_bg, V_bg):
        if K_bg.shape[0] != key.shape[0]:
            r = key.shape[0] // K_bg.shape[0]
            K_bg, V_bg = K_bg.repeat(r, 1, 1), V_bg.repeat(r, 1, 1)
        return K_bg, V_bg

    def _subsample_kv(self, layer_id, K, V): return K, V

    def _sdpa(self, q, k, v, scale=None):
        if scale: q = q * scale * math.sqrt(q.shape[-1])
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    def _manual_attn(self, q, k, v, scale):
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
        attn_probs = attn_scores.softmax(dim=-1)
        context = torch.bmm(attn_probs, v)
        return context, attn_probs

    # 【新增】生成当前层的查询掩码
    def _make_query_mask(self, layer_id: str, q: torch.Tensor) -> Optional[torch.Tensor]:
        if self._region_mask is None or self.base_latent_hw is None: return None
        H0, W0 = self.base_latent_hw
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, H0 // s), max(1, W0 // s)
        Lq = q.shape[1]
        
        # 仅当序列长度匹配时生效 (Cross-Attn Q is spatial)
        if Lq != h * w: return None
        
        # 下采样 mask
        m = F.interpolate(self._region_mask, size=(h, w), mode="bilinear", align_corners=False)
        m1d = m.reshape(1, 1, -1) # [1,1,Lq]
        
        # 扩展到 Batch*Heads
        BxH = q.shape[0]
        m1d = m1d.expand(BxH, -1, -1).squeeze(1) # [BxH, Lq]
        return m1d.clamp(0.0, 1.0)

    def _probe_cross_attn(self, attn_probs, layer_id, batch_size):
        if not (self.probe_enabled and self.layer_selector.for_probe(layer_id) and self.subject_id is not None): return
        token_ids = self.subject_token_ids.get(self.subject_id, [])
        if not token_ids: return
        idx = torch.tensor(token_ids, dtype=torch.long, device=attn_probs.device)
        sel = attn_probs.index_select(dim=-1, index=idx).mean(dim=-1)
        bxh, Lq = sel.shape
        heads = bxh // max(1, batch_size)
        sel = sel.reshape(batch_size, heads, Lq).mean(dim=1)
        amap_1d = sel[0:1] 
        if self.base_latent_hw is None: return
        h0, w0 = self.base_latent_hw
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, h0 // s), max(1, w0 // s)
        if amap_1d.shape[-1] != h * w:
            sq = int(math.sqrt(amap_1d.shape[-1]))
            if sq*sq == amap_1d.shape[-1]: h, w = sq, sq
            else: return
        amap_2d = amap_1d.reshape(1, 1, h, w)
        amap_up = F.interpolate(amap_2d, size=(h0, w0), mode="bilinear", align_corners=False)
        prev = self._attn_accumulator.get(self.subject_id, None)
        self._attn_accumulator[self.subject_id] = amap_up if prev is None else (prev + amap_up)

    # --- Forward ---
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        layer_id = getattr(attn, "layer_id", None)
        is_self_attn = encoder_hidden_states is None
        
        q = attn.to_q(hidden_states)
        if is_self_attn:
            k, v = attn.to_k(hidden_states), attn.to_v(hidden_states)
        else:
            k, v = attn.to_k(encoder_hidden_states), attn.to_v(encoder_hidden_states)

        q, k, v = attn.head_to_batch_dim(q), attn.head_to_batch_dim(k), attn.head_to_batch_dim(v)

        # Self-Attn Injection
        if is_self_attn and self.layer_selector.for_inject(layer_id):
            if self.mode == "record_bg":
                self.context_bank.put(layer_id, k, v)
            elif self.mode == "inject_subject" and self.context_bank.has(layer_id):
                K_bg, V_bg = self.context_bank.get(layer_id)
                K_bg, V_bg = self._align_bg_batch(k, K_bg.to(k), V_bg.to(v))
                alpha = self.alpha_scheduler.alpha(self.step_idx, self.total_steps, layer_id)
                if alpha > 0:
                    k, v = torch.cat([k, alpha*K_bg], 1), torch.cat([v, alpha*V_bg], 1)
                attention_mask = None

        scale = getattr(attn, "scale", 1.0 / math.sqrt(q.shape[-1]))

        if is_self_attn:
            context = self._sdpa(q, k, v, scale)
        else:
            # Cross-Attn Logic
            # 【关键修改】如果是在做 Subject Injection 并且层允许，就启用门控
            # 注意：LayerSelector 这里既控制 Self-Attn 注入，也控制 Cross-Attn 门控，这是合理的
            if self.mode == "inject_subject" and self.layer_selector.for_inject(layer_id):
                # 必须用 Manual Attn 才能修改 probs
                context, attn_probs = self._manual_attn(q, k, v, scale)
                
                # 1. 探针 (如果启用)
                self._probe_cross_attn(attn_probs.detach(), layer_id, hidden_states.shape[0])
                
                # 2. 空间门控 (Spatial Gating)
                qm = self._make_query_mask(layer_id, q)
                if qm is not None:
                    # qm: [B*H, Lq] -> unsqueeze -> [B*H, Lq, 1] (广播到 Key 维)
                    attn_probs = attn_probs * qm.unsqueeze(-1)
                    # 重新归一化
                    denom = attn_probs.sum(dim=-1, keepdim=True) + 1e-6
                    attn_probs = attn_probs / denom
                    context = torch.bmm(attn_probs, v)
            else:
                # 默认走 SDPA
                context = self._sdpa(q, k, v, scale)

        return attn.to_out[0](attn.batch_to_head_dim(context))

def create_carc_attention_processor(context_bank=None, alpha_scheduler=None, layer_selector=None):
    return CARCAttentionProcessor(context_bank, alpha_scheduler, layer_selector)
