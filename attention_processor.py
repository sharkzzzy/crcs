"""
core/attention_processor.py
FINAL V5: User-Optimized Edition.
Features: SDPA, KV-Subsampling, Float16 Storage, Strict Layer Filtering.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Context Bank (Float16 Optimized)
# ==============================================================================
class ContextBank:
    def __init__(self):
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}

    def clear(self):
        self._store.clear()

    def put(self, layer_id: str, K: torch.Tensor, V: torch.Tensor):
        # 强制转半精度 + detach，极大节省显存
        self._store[layer_id] = {
            "K": K.detach().to(dtype=torch.float16),
            "V": V.detach().to(dtype=torch.float16),
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
        self.per_layer_scaling = per_layer_scaling or {"lowres": 1.0, "midres": 0.6, "hires": 0.3}

    def alpha(self, step_idx: int, total_steps: int, layer_id: str) -> float:
        frac = step_idx / max(total_steps - 1, 1)
        val = self.phases[-1][2] # Default to last phase
        for start, end, a in self.phases:
            if start <= frac <= end:
                val = a
                break

        lid = layer_id or ""
        # 简单层级判断
        if "down_blocks.2" in lid or "mid_block" in lid or "up_blocks.0" in lid:
            scale = self.per_layer_scaling.get("midres", 0.6)
        elif "down_blocks.0" in lid or "down_blocks.1" in lid:
            scale = self.per_layer_scaling.get("lowres", 1.0)
        else:
            scale = self.per_layer_scaling.get("hires", 0.3)
        
        return float(val * scale)


# ==============================================================================
# 3. Layer Selector
# ==============================================================================
class LayerSelector:
    def __init__(self, patterns=None, probe_patterns=None):
        # 修正：SDXL中 up_blocks.0 是深层(32x32)，up_blocks.2 是浅层(128x128)
        # 默认只在 32x32 层注入，最安全
        self.patterns = patterns or ["down_blocks.2", "mid_block", "up_blocks.0"]
        # 探针也只在最核心层收集
        self.probe_patterns = probe_patterns or ["mid_block"]

    def for_inject(self, layer_id: str) -> bool:
        return layer_id and any(p in layer_id for p in self.patterns)

    def for_probe(self, layer_id: str) -> bool:
        return layer_id and any(p in layer_id for p in self.probe_patterns)


def _infer_scale_from_layer_id(layer_id: str) -> int:
    if not layer_id: return 1
    if "mid_block" in layer_id: return 8
    m = re.search(r"down_blocks.(\d+)", layer_id)
    if m: return 2 ** int(m.group(1)) # down.0=1(128), down.1=2(64), down.2=4(32)
    m2 = re.search(r"up_blocks.(\d+)", layer_id)
    if m2:
        idx = int(m2.group(1)) 
        # SDXL UpBlock mapping: up.0(32)->4, up.1(64)->2, up.2(128)->1
        mapping = {0: 4, 1: 2, 2: 1}
        return mapping.get(idx, 1)
    return 1


# ==============================================================================
# 4. Attention Processor (High Performance)
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

    @torch.no_grad()
    def pop_attn_maps(self) -> Dict[int, torch.Tensor]:
        out = {}
        for sid, amap in self._attn_accumulator.items():
            mi, ma = amap.min(), amap.max()
            out[sid] = (amap - mi) / (ma - mi + 1e-8)
        self._attn_accumulator.clear()
        return out

    # --- Helpers ---
    def _align_bg_batch(self, key: torch.Tensor, K_bg: torch.Tensor, V_bg: torch.Tensor):
        if K_bg.shape[0] != key.shape[0]:
            ratio = key.shape[0] // K_bg.shape[0]
            if ratio > 1:
                K_bg = K_bg.repeat(ratio, 1, 1)
                V_bg = V_bg.repeat(ratio, 1, 1)
        return K_bg, V_bg

    def _subsample_kv(self, layer_id: str, K: torch.Tensor, V: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggressive optimization: Average Pool K/V to reduce sequence length L.
        """
        if self.base_latent_hw is None: return K, V
        
        H0, W0 = self.base_latent_hw
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, H0 // s), max(1, W0 // s)
        L = K.shape[1]
        
        # Only subsample if dimensions imply spatial layout
        if L != h * w: return K, V

        # Dynamic stride based on resolution
        if h >= 64 or w >= 64: stride = 4
        elif h >= 32 or w >= 32: stride = 2
        else: stride = 1
        
        if stride == 1: return K, V

        # Reshape [B*H, L, D] -> [B*H, D, h, w] for pooling
        BxH, _, Dh = K.shape
        K2 = K.view(BxH, h, w, Dh).permute(0, 3, 1, 2)
        V2 = V.view(BxH, h, w, Dh).permute(0, 3, 1, 2)
        
        K2 = F.avg_pool2d(K2, kernel_size=stride, stride=stride)
        V2 = F.avg_pool2d(V2, kernel_size=stride, stride=stride)
        
        h2, w2 = K2.shape[-2], K2.shape[-1]
        
        # Flatten back
        Kd = K2.permute(0, 2, 3, 1).reshape(BxH, h2 * w2, Dh)
        Vd = V2.permute(0, 2, 3, 1).reshape(BxH, h2 * w2, Dh)
        return Kd.contiguous(), Vd.contiguous()

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = None) -> torch.Tensor:
        """
        Flash Attention via PyTorch SDPA.
        Ignores mask for Self-Attn to prevent broadcasting explosion.
        """
        # Manual scale Q if needed (SDPA default is 1/sqrt(D))
        # Diffusers sometimes uses custom scale, but usually it matches default.
        if scale is not None:
             # Re-scale Q to match SDPA expectation if it differs, 
             # but here we trust SDPA's internal scaling or apply before.
             # Safe bet: Apply scale manually to Q and rely on SDPA
             q = q * scale * math.sqrt(q.shape[-1])
             
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    def _manual_attn(self, q, k, v, scale):
        # Fallback for Cross-Attn Probe (need weights)
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
        sel = attn_probs.index_select(dim=-1, index=idx).mean(dim=-1) # [B*H, Lq]
        
        bxh, Lq = sel.shape
        heads = bxh // max(1, batch_size)
        sel = sel.reshape(batch_size, heads, Lq).mean(dim=1)
        amap_1d = sel[0:1] # Take first batch item (Cond)

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

    # --- Forward ---
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
                # 1. Device/Dtype Align
                K_bg = K_bg.to(device=k.device, dtype=k.dtype)
                V_bg = V_bg.to(device=v.device, dtype=v.dtype)
                # 2. Batch Align
                K_bg, V_bg = self._align_bg_batch(k, K_bg, V_bg)
                # 3. Subsample (Optional optimization)
                K_bg, V_bg = self._subsample_kv(layer_id, K_bg, V_bg)
                
                # 4. Inject
                alpha_val = self.alpha_scheduler.alpha(self.step_idx, self.total_steps, layer_id)
                if alpha_val > 0.0:
                    k = torch.cat([k, alpha_val * K_bg], dim=1)
                    v = torch.cat([v, alpha_val * V_bg], dim=1)
                
                # 5. MASK POLICY: DROP IT.
                # In SDXL Self-Attn, dropping mask is safe and prevents 1280GB broadcast errors.
                attention_mask = None

        # Compute
        head_dim = q.shape[-1]
        scale = getattr(attn, "scale", 1.0 / math.sqrt(head_dim))

        if is_self_attn:
            # Always SDPA for Self-Attn (Speed + No OOM)
            context = self._sdpa(q, k, v, scale)
        else:
            # Cross-Attn: Manual only if probing
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
