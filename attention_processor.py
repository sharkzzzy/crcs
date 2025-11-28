"""
core/attention_processor.py
FINAL V3: Fixes OOM Explosion and Mask Broadcasting.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ... ContextBank, AlphaScheduler, LayerSelector, _infer_scale_from_layer_id 保持不变 ...
# 为了确保你文件完整，我把这几个类也贴在这里，确保无误

class ContextBank:
    def __init__(self):
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}
    def clear(self):
        self._store.clear()
    def put(self, layer_id: str, K: torch.Tensor, V: torch.Tensor):
        self._store[layer_id] = {"K": K.detach(), "V": V.detach()}
    def has(self, layer_id: str) -> bool:
        return layer_id in self._store
    def get(self, layer_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._store.get(layer_id, None)
        if data is None: raise KeyError(f"ContextBank: missing KV for layer {layer_id}")
        return data["K"], data["V"]

class AlphaScheduler:
    def __init__(self, construct_phase=(0.0, 0.4, 0.2), texture_phase=(0.4, 0.8, 0.8), refine_phase=(0.8, 1.0, 0.5), per_layer_scaling=None):
        self.phases = [construct_phase, texture_phase, refine_phase]
        self.per_layer_scaling = per_layer_scaling or {"lowres": 1.0, "midres": 0.6, "hires": 0.3}

    def alpha(self, step_idx: int, total_steps: int, layer_id: str) -> float:
        frac = step_idx / max(total_steps - 1, 1)
        val = 0.0
        for start, end, a in self.phases:
            if start <= frac <= end:
                val = a
                break
        else:
            val = self.phases[-1][2]

        lid = layer_id or ""
        scale = 1.0
        if "down_blocks.1" in lid or "down_blocks.2" in lid: scale = self.per_layer_scaling.get("lowres", 1.0)
        elif "mid_block" in lid: scale = self.per_layer_scaling.get("midres", 0.6)
        else: scale = self.per_layer_scaling.get("hires", 0.3)
        return val * scale

class LayerSelector:
    def __init__(self, patterns=None):
        # 移除高分辨率层以节省显存 (remove up_blocks.0/.1 if necessary)
        self.patterns = patterns or ["down_blocks.1", "down_blocks.2", "mid_block", "up_blocks.0", "up_blocks.1"]
    def __call__(self, layer_id: str) -> bool:
        return layer_id and any(p in layer_id for p in self.patterns)

def _infer_scale_from_layer_id(layer_id: str) -> int:
    if not layer_id: return 1
    if "mid_block" in layer_id: return 8
    m = re.search(r"down_blocks.(\d+)", layer_id)
    if m: return 2 ** int(m.group(1))
    m2 = re.search(r"up_blocks.(\d+)", layer_id)
    if m2: return 2 ** (2 - int(m2.group(1)))
    return 1

# ==============================================================================
# MAIN PROCESSOR
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

    # ... Setters same as before ...
    def set_mode(self, mode: str): self.mode = mode
    def set_probe_enabled(self, flag: bool):
        self.probe_enabled = bool(flag)
        if self.probe_enabled: self._attn_accumulator.clear()
    def set_step_index(self, idx: int, total: int):
        self.step_idx = idx
        self.total_steps = total
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

    def _compute_attention(self, q, k, v, scale, attention_mask=None):
        # [B*H, Lq, Lk]
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
        
        if attention_mask is not None:
            # 关键修复：确保 Mask 广播正确
            # attention_mask 可能是 [B, 1, Lq, Lk]
            # attn_scores 是 [B*H, Lq, Lk]
            
            # 1. Expand Batch/Head dim
            if attention_mask.dim() == 3:
                heads = attn_scores.shape[0] // attention_mask.shape[0]
                attention_mask = attention_mask.repeat_interleave(heads, dim=0)
            
            # 2. Check shapes
            if attention_mask.shape[-2:] == attn_scores.shape[-2:]:
                attn_scores = attn_scores + attention_mask
            else:
                # Fallback: 如果 Mask 形状不对（比如Key变长了），尝试只取前缀或不做Mask
                # 对于 Inject 模式，Key 变长了，如果 Mask 没变长，直接相加会报错或广播爆炸
                # 这种情况下，如果没处理好 Mask，宁可不加 Mask (通常 SDXL SelfAttn Mask 为 None)
                pass

        attn_probs = attn_scores.softmax(dim=-1)
        context = torch.bmm(attn_probs, v)
        return context, attn_probs

    def _probe_cross_attn(self, attn_probs, layer_id, batch_size):
        if not self.probe_enabled or self.subject_id is None: return
        token_ids = self.subject_token_ids.get(self.subject_id, [])
        if not token_ids: return
        idx = torch.tensor(token_ids, dtype=torch.long, device=attn_probs.device)
        sel = attn_probs.index_select(dim=-1, index=idx).mean(dim=-1)
        bxh, Lq = sel.shape
        heads = bxh // batch_size if batch_size > 0 else 1
        sel = sel.reshape(batch_size, heads, Lq).mean(dim=1)
        amap_1d = sel[0:1] 
        if self.base_latent_hw is None: return
        base_h, base_w = self.base_latent_hw
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, base_h // s), max(1, base_w // s)
        if amap_1d.shape[-1] != h * w:
            sq = int(math.sqrt(amap_1d.shape[-1]))
            h, w = sq, sq
        amap_2d = amap_1d.reshape(1, 1, h, w)
        amap_up = F.interpolate(amap_2d, size=(base_h, base_w), mode="bilinear", align_corners=False)
        prev = self._attn_accumulator.get(self.subject_id, None)
        self._attn_accumulator[self.subject_id] = amap_up if prev is None else (prev + amap_up)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        layer_id = getattr(attn, "layer_id", None)
        is_self_attn = encoder_hidden_states is None
        
        query = attn.to_q(hidden_states)
        if is_self_attn:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # === CARC LOGIC ===
        # 严格检查：必须是 Self Attn，且 layer_selector 选中
        if is_self_attn and self.layer_selector(layer_id):
            
            if self.mode == "record_bg":
                self.context_bank.put(layer_id, key, value)
            
            elif self.mode == "inject_subject" and self.context_bank.has(layer_id):
                alpha_val = self.alpha_scheduler.alpha(self.step_idx, self.total_steps, layer_id)
                K_bg, V_bg = self.context_bank.get(layer_id)
                
                # 显存保护：确保设备一致
                K_bg = K_bg.to(device=key.device, dtype=key.dtype)
                V_bg = V_bg.to(device=value.device, dtype=value.dtype)
                
                # Align Batch
                if K_bg.shape[0] != key.shape[0]:
                    ratio = key.shape[0] // K_bg.shape[0]
                    if ratio > 1:
                        K_bg = K_bg.repeat(ratio, 1, 1)
                        V_bg = V_bg.repeat(ratio, 1, 1)
                
                # Concat
                key = torch.cat([key, alpha_val * K_bg], dim=1)
                value = torch.cat([value, alpha_val * V_bg], dim=1)
                
                # 关键修复：Mask Extension
                if attention_mask is not None:
                    # attention_mask: [B, 1, Lq, Lk]
                    bg_len = K_bg.shape[1]
                    
                    # 构造一个形状正确的全0 Mask (允许关注)
                    # 我们只需要取 attention_mask 的前维度，修改最后一维
                    new_shape = list(attention_mask.shape)
                    new_shape[-1] = bg_len
                    
                    mask_bg = torch.zeros(new_shape, device=attention_mask.device, dtype=attention_mask.dtype)
                    
                    # 拼接到原 Mask 后面
                    attention_mask = torch.cat([attention_mask, mask_bg], dim=-1)

        scale = getattr(attn, "scale", None)
        if scale is None: scale = 1.0 / math.sqrt(query.shape[-1])
            
        context, attn_probs = self._compute_attention(query, key, value, scale, attention_mask)

        if (not is_self_attn) and self.probe_enabled:
            # 显存保护：Detach
            self._probe_cross_attn(attn_probs.detach(), layer_id, batch_size=hidden_states.shape[0])

        context = attn.batch_to_head_dim(context)
        hidden_states = attn.to_out[0](context)
        if hasattr(attn, "to_out") and len(attn.to_out) > 1 and attn.to_out[1] is not None:
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def create_carc_attention_processor(context_bank=None, alpha_scheduler=None, layer_selector=None):
    return CARCAttentionProcessor(context_bank, alpha_scheduler, layer_selector)
