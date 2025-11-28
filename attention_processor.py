"""
core/attention_processor.py
FINAL V9: Full Float32 Edition (No casting to half precision).
Features: SDPA, High-Res Injection, Strict Batch Alignment.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Context Bank (FP32)
# ==============================================================================
class ContextBank:
    def __init__(self):
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}

    def clear(self):
        self._store.clear()

    def put(self, layer_id: str, K: torch.Tensor, V: torch.Tensor):
        # 仅 detach，保持原始精度 (FP32)
        self._store[layer_id] = {
            "K": K.detach(),
            "V": V.detach(),
        }

    def has(self, layer_id: str) -> bool:
        return layer_id in self._store

    def get(self, layer_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._store.get(layer_id, None)
        if data is None:
            raise KeyError(f"ContextBank: missing KV for layer {layer_id}")
        return data["K"], data["V"]


# ==============================================================================
# 2. Alpha Scheduler
# ==============================================================================
class AlphaScheduler:
    def __init__(
        self, 
        construct_phase=(0.0, 0.4, 0.2), 
        texture_phase=(0.4, 0.8, 0.8), 
        refine_phase=(0.8, 1.0, 0.5), 
        per_layer_scaling=None
    ):
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
        scale = 1.0
        
        if "down_blocks.2" in lid or "mid_block" in lid or "up_blocks.0" in lid:
            scale = self.per_layer_scaling.get("midres", 0.6)
        elif "down_blocks.0" in lid or "down_blocks.1" in lid:
            scale = self.per_layer_scaling.get("lowres", 1.0)
        else:
            if is_refine:
                scale = self.per_layer_scaling.get("hires", 0.35)
            else:
                scale = 0.0
        
        return float(val * scale)


# ==============================================================================
# 3. Layer Selector
# ==============================================================================
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
    if m2:
        return {0: 4, 1: 2, 2: 1}.get(int(m2.group(1)), 1)
    return 1


# ==============================================================================
# 4. Attention Processor (FP32)
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

    def set_mode(self, mode: str): self.mode = mode
    def set_probe_enabled(self, flag: bool):
        self.probe_enabled = bool(flag)
        if self.probe_enabled: self._attn_accumulator.clear()
    def set_step_index(self, idx: int, total: int):
        self.step_idx, self.total_steps = int(idx), int(total)
    def set_subject_id(self, i: Optional[int]): self.subject_id = i
    def set_subject_token_ids(self, mapping: Dict[int, List[int]]): self.subject_token_ids = mapping or {}
    def set_base_latent_hw(self, h: int, w: int): self.base_latent_hw = (int(h), int(w))

    @torch.no_grad()
    def pop_attn_maps(self) -> Dict[int, torch.Tensor]:
        out = {}
        for sid, amap in self._attn_accumulator.items():
            mi, ma = amap.min(), amap.max()
            out[sid] = (amap - mi) / (ma - mi + 1e-8)
        self._attn_accumulator.clear()
        return out

    def _align_bg_batch(self, key: torch.Tensor, K_bg: torch.Tensor, V_bg: torch.Tensor):
        if K_bg.shape[0] != key.shape[0]:
            ratio = key.shape[0] // K_bg.shape[0]
            if ratio > 1:
                K_bg = K_bg.repeat(ratio, 1, 1)
                V_bg = V_bg.repeat(ratio, 1, 1)
        return K_bg, V_bg

    def _subsample_kv(self, layer_id: str, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return K, V

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = None) -> torch.Tensor:
        if scale is not None:
             q = q * scale * math.sqrt(q.shape[-1])
        # SDPA supports float32 natively
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    def _manual_attn(self, q, k, v, scale):
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
        attn_probs = attn_scores.softmax(dim=-1)
        context = torch.bmm(attn_probs, v)
        return context, attn_probs

    def _probe_cross_attn(self, attn_probs, layer_id, batch_size):
        if not (self.probe_enabled and self.layer_selector.for_probe(layer_id) and self.subject_id is not None):
            return
        token_ids = self.subject_token_ids.get(self.subject_id, [])
        if not token_ids: return
        
        idx = torch.tensor(token_ids, dtype=torch.long, device=attn_probs.device)
        sel = attn_probs.index_select(dim=-1, index=idx).mean(dim=-1)
        
        bxh, Lq = sel.shape
        heads = bxh // max(1, batch_size)
        sel = sel.reshape(batch_size, heads, Lq).mean(dim=1)
        amap_1d = sel[0:1]

        if self.base_latent_hw is None: return
        base_h, base_w = self.base_latent_hw
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, base_h // s), max(1, base_w // s)
        
        if amap_1d.shape[-1] != h * w:
            sq = int(math.sqrt(amap_1d.shape[-1]))
            if sq*sq == amap_1d.shape[-1]: h, w = sq, sq
            else: return

        amap_2d = amap_1d.reshape(1, 1, h, w)
        amap_up = F.interpolate(amap_2d, size=(base_h, base_w), mode="bilinear", align_corners=False)
        
        prev = self._attn_accumulator.get(self.subject_id, None)
        self._attn_accumulator[self.subject_id] = amap_up if prev is None else (prev + amap_up)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        layer_id = getattr(attn, "layer_id", None)
        is_self_attn = encoder_hidden_states is None
        
        q = attn.to_q(hidden_states)
        if is_self_attn:
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
        else:
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        # === CARC SELF-ATTN LOGIC ===
        if is_self_attn and self.layer_selector.for_inject(layer_id):
            
            if self.mode == "record_bg":
                self.context_bank.put(layer_id, k, v)
                
            elif self.mode == "inject_subject" and self.context_bank.has(layer_id):
                K_bg, V_bg = self.context_bank.get(layer_id)
                
                # 精度对齐 (FP32)
                K_bg = K_bg.to(device=k.device, dtype=k.dtype)
                V_bg = V_bg.to(device=v.device, dtype=v.dtype)
                
                K_bg, V_bg = self._align_bg_batch(k, K_bg, V_bg)
                K_bg, V_bg = self._subsample_kv(layer_id, K_bg, V_bg)
                
                alpha_val = self.alpha_scheduler.alpha(self.step_idx, self.total_steps, layer_id)
                if alpha_val > 0.0:
                    k = torch.cat([k, alpha_val * K_bg], dim=1)
                    v = torch.cat([v, alpha_val * V_bg], dim=1)
                
                attention_mask = None

        scale = getattr(attn, "scale", 1.0 / math.sqrt(head_dim)) if 'head_dim' in locals() else getattr(attn, "scale", 1.0 / math.sqrt(q.shape[-1]))

        if is_self_attn:
            context = self._sdpa(q, k, v, scale)
        else:
            if self.probe_enabled and self.layer_selector.for_probe(layer_id):
                context, attn_probs = self._manual_attn(q, k, v, scale)
                self._probe_cross_attn(attn_probs.detach(), layer_id, batch_size=hidden_states.shape[0])
            else:
                context = self._sdpa(q, k, v, scale)

        context = attn.batch_to_head_dim(context)
        hidden_states = attn.to_out[0](context)
        if hasattr(attn, "to_out") and len(attn.to_out) > 1 and attn.to_out[1] is not None:
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def create_carc_attention_processor(context_bank=None, alpha_scheduler=None, layer_selector=None):
    return CARCAttentionProcessor(context_bank, alpha_scheduler, layer_selector)



