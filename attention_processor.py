import math
import re
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Context Bank (KV Cache)
# ==============================================================================
class ContextBank:
    """
    Stores background K/V per attention layer (self-attn only).
    Keys are layer names (strings). Values are dict with 'K' and 'V' tensors.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}

    def clear(self):
        self._store.clear()

    def put(self, layer_id: str, K: torch.Tensor, V: torch.Tensor):
        # Store clones to avoid reference issues; detach to save memory/graph
        self._store[layer_id] = {"K": K.detach(), "V": V.detach()}

    def has(self, layer_id: str) -> bool:
        return layer_id in self._store

    def get(self, layer_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._store.get(layer_id, None)
        if data is None:
            raise KeyError(f"ContextBank: missing KV for layer {layer_id}")
        return data["K"], data["V"]


# ==============================================================================
# 2. Alpha Scheduler (Dynamic Injection Weight)
# ==============================================================================
class AlphaScheduler:
    """
    Time-dependent alpha(t) for background KV injection.
    Supports per-layer scaling (low/mid/hi) inferred from layer_id.
    """
    def __init__(
        self,
        construct_phase=(0.7, 1.0, 0.2), # (start_frac, end_frac, alpha) - Early
        texture_phase=(0.2, 0.7, 1.0),   # - Mid (Strong Injection)
        refine_phase=(0.0, 0.2, 0.5),    # - Late
        per_layer_scaling: Dict[str, float] = None
    ):
        self.construct_phase = construct_phase
        self.texture_phase = texture_phase
        self.refine_phase = refine_phase
        self.per_layer_scaling = per_layer_scaling or {"lowres": 1.0, "midres": 0.6, "hires": 0.3}

    def _phase_alpha(self, t: int, T: int) -> float:
        # Interpret t as remaining steps (e.g. 50 -> 0).
        # frac goes from 1.0 (start) to 0.0 (end) roughly.
        frac = t / max(T - 1, 1)
        
        s1, e1, a1 = self.construct_phase
        s2, e2, a2 = self.texture_phase
        s3, e3, a3 = self.refine_phase
        
        if s1 <= frac <= e1: return a1
        if s2 <= frac <= e2: return a2
        if s3 <= frac <= e3: return a3
        
        # Fallback for gaps
        if frac > e1: return a1
        if frac < s3: return a3
        return a2

    def _layer_scale(self, layer_id: str) -> float:
        lid = layer_id or ""
        if "down_blocks.0" in lid or "down_blocks.1" in lid:
            return self.per_layer_scaling.get("lowres", 1.0)
        if "mid_block" in lid or "down_blocks.2" in lid or "up_blocks.2" in lid:
            return self.per_layer_scaling.get("midres", 0.6)
        return self.per_layer_scaling.get("hires", 0.3)

    def alpha(self, t: int, T: int, layer_id: Optional[str]) -> float:
        base = self._phase_alpha(t, T)
        scale = self._layer_scale(layer_id or "")
        return base * scale


# ==============================================================================
# 3. Layer Selector
# ==============================================================================
class LayerSelector:
    """
    Decides which layers apply KV injection/recording.
    Default: Deep layers only (mid_block + deep down/up).
    """
    def __init__(self, patterns: Optional[List[str]] = None):
        # Default SDXL critical layers for structure
        self.patterns = patterns or ["down_blocks.1", "down_blocks.2", "mid_block", "up_blocks.0", "up_blocks.1"]

    def __call__(self, layer_id: str) -> bool:
        if layer_id is None:
            return False
        return any(p in layer_id for p in self.patterns)


def _infer_scale_from_layer_id(layer_id: str) -> int:
    """Helper to guess resolution scale factor relative to latent."""
    if layer_id is None: return 1
    # Simple heuristic for SDXL
    if "mid_block" in layer_id: return 8
    # down.0->1, down.1->2, down.2->4
    m = re.search(r"down_blocks.(\d+)", layer_id)
    if m: return 2 ** int(m.group(1))
    # up.0->4, up.1->2, up.2->1 (Approximate mapping for SDXL Base)
    m2 = re.search(r"up_blocks.(\d+)", layer_id)
    if m2: return 2 ** (2 - int(m2.group(1))) # This varies by model, but roughly ok
    return 1


# ==============================================================================
# 4. Main Processor
# ==============================================================================
class CARCAttentionProcessor(nn.Module):
    """
    A drop-in AttentionProcessor implementing CARC logic.
    """
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

        self.mode: str = "off"  # 'off' | 'record_bg' | 'inject_subject'
        self.probe_enabled: bool = False
        
        # State variables updated during loop
        self.t: Optional[int] = None
        self.T: Optional[int] = None
        self.subject_id: Optional[int] = None

        self.base_latent_hw: Optional[Tuple[int, int]] = None
        self.subject_token_ids: Dict[int, List[int]] = {}
        self._attn_accumulator: Dict[int, torch.Tensor] = {}

    # --- Configuration APIs ---
    def set_mode(self, mode: str):
        assert mode in ("off", "record_bg", "inject_subject")
        self.mode = mode

    def set_probe_enabled(self, flag: bool):
        self.probe_enabled = bool(flag)
        if self.probe_enabled:
            self._attn_accumulator.clear()

    def set_step(self, t: int, T: int):
        self.t, self.T = int(t), int(T)

    def set_subject_id(self, i: Optional[int]):
        self.subject_id = None if i is None else int(i)

    def set_subject_token_ids(self, mapping: Dict[int, List[int]]):
        self.subject_token_ids = mapping or {}

    def set_base_latent_hw(self, h: int, w: int):
        self.base_latent_hw = (int(h), int(w))

    @torch.no_grad()
    def pop_attn_maps(self) -> Dict[int, torch.Tensor]:
        """
        Returns normalized accumulated attention maps.
        """
        out = {}
        for sid, amap in self._attn_accumulator.items():
            # Min-Max Normalize
            mi, ma = amap.min(), amap.max()
            if ma - mi > 1e-6:
                a = (amap - mi) / (ma - mi)
            else:
                a = amap - mi
            out[sid] = a
        self._attn_accumulator.clear()
        return out

    # --- Internal Helpers ---
    def _compute_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        scale: float, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # q,k,v: [B*H, Seq, Dim]
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * scale
        
        if attention_mask is not None:
            # Handle broadcasting
            # Mask input: [B, 1, Lk] or [B, Lq, Lk]
            if attention_mask.dim() == 3:
                # Expand head dim to match B*H
                heads = attn_scores.shape[0] // attention_mask.shape[0]
                attention_mask = attention_mask.repeat_interleave(heads, dim=0)
            
            # Add mask (assuming 0 is keep, -inf is drop, or standard SD mask)
            # Ensure shape match
            if attention_mask.shape[-2:] == attn_scores.shape[-2:]:
                attn_scores = attn_scores + attention_mask
        
        attn_probs = attn_scores.softmax(dim=-1)
        context = torch.bmm(attn_probs, v)
        return context, attn_probs

    def _probe_cross_attn(self, attn_probs: torch.Tensor, layer_id: str, batch_size: int):
        """Aggregate attention weights for specific tokens."""
        if not self.probe_enabled or self.subject_id is None:
            return
        token_ids = self.subject_token_ids.get(self.subject_id, None)
        if not token_ids:
            return

        # attn_probs: [B*H, Lq, Lk]
        # Gather by key index (token_ids)
        idx = torch.tensor(token_ids, dtype=torch.long, device=attn_probs.device)
        
        # [B*H, Lq, n_tokens] -> mean -> [B*H, Lq]
        sel = attn_probs.index_select(dim=-1, index=idx).mean(dim=-1)
        
        # Reshape to [B, H, Lq] and mean over heads -> [B, Lq]
        bxh, Lq = sel.shape
        heads = bxh // batch_size if batch_size > 0 else 1
        sel = sel.reshape(batch_size, heads, Lq).mean(dim=1)
        
        # Assume batch=1 for probing usually
        amap_1d = sel # [B, Lq]
        
        if self.base_latent_hw is None: return
        base_h, base_w = self.base_latent_hw
        
        # Heuristic reshape to 2D
        s = _infer_scale_from_layer_id(layer_id)
        h, w = max(1, base_h // s), max(1, base_w // s)
        
        # Fallback if dimensions don't match
        if amap_1d.shape[-1] != h * w:
             # Try simple square root
            sq = int(math.sqrt(amap_1d.shape[-1]))
            h, w = sq, sq
            
        amap_2d = amap_1d.reshape(batch_size, 1, h, w)
        
        # Upsample to common resolution for accumulation
        amap_up = F.interpolate(
            amap_2d, size=(base_h, base_w), mode="bilinear", align_corners=False
        )
        
        prev = self._attn_accumulator.get(self.subject_id, None)
        self._attn_accumulator[self.subject_id] = amap_up if prev is None else (prev + amap_up)

    # --- Forward Pass ---
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args, **kwargs
    ):
        layer_id = getattr(attn, "layer_id", None)
        is_self_attn = encoder_hidden_states is None
        
        # 1. Projections
        query = attn.to_q(hidden_states)
        
        if is_self_attn:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        # 2. Reshape to Heads [B, L, D] -> [B*H, L, D_head]
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 3. CARC Logic: Self-Attn Injection / Recording
        if is_self_attn and self.layer_selector(layer_id):
            
            # A) Record Mode
            if self.mode == "record_bg":
                self.context_bank.put(layer_id, key, value)
            
            # B) Inject Mode
            elif self.mode == "inject_subject" and self.context_bank.has(layer_id):
                alpha_val = self.alpha_scheduler.alpha(self.t or 0, self.T or 1, layer_id)
                
                K_bg, V_bg = self.context_bank.get(layer_id)
                K_bg = K_bg.to(key.dtype)
                V_bg = V_bg.to(value.dtype)
                
                # Concat: [Self, alpha*BG]
                # Note: Concatenating on dim=1 (Sequence Length)
                key = torch.cat([key, alpha_val * K_bg], dim=1)
                value = torch.cat([value, alpha_val * V_bg], dim=1)
                
                # Fix: Extend attention_mask if present
                if attention_mask is not None:
                    # attention_mask is typically [B, 1, L_self] or [B, L_self, L_self]
                    # We need to pad it to match [..., L_self + L_bg]
                    bg_len = K_bg.shape[1]
                    
                    # Create zero-mask (allow attention) for the appended bg tokens
                    # Shape: match batch and query dims of original mask
                    padding_shape = list(attention_mask.shape)
                    padding_shape[-1] = bg_len
                    
                    mask_bg = torch.zeros(
                        padding_shape, 
                        device=attention_mask.device, 
                        dtype=attention_mask.dtype
                    )
                    attention_mask = torch.cat([attention_mask, mask_bg], dim=-1)

        # 4. Compute Attention
        scale = getattr(attn, "scale", None)
        if scale is None:
            scale = 1.0 / math.sqrt(query.shape[-1])
            
        context, attn_probs = self._compute_attention(query, key, value, scale, attention_mask)

        # 5. CARC Logic: Cross-Attn Probing
        if (not is_self_attn) and self.probe_enabled:
            # Important: detach to avoid VRAM leak
            self._probe_cross_attn(attn_probs.detach(), layer_id, batch_size=hidden_states.shape[0])

        # 6. Output Projection
        context = attn.batch_to_head_dim(context)
        hidden_states = attn.to_out[0](context)
        if hasattr(attn, "to_out") and len(attn.to_out) > 1 and attn.to_out[1] is not None:
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

# Helper factory
def create_carc_attention_processor(
    context_bank: Optional[ContextBank] = None,
    alpha_scheduler: Optional[AlphaScheduler] = None,
    layer_selector: Optional[LayerSelector] = None,
) -> CARCAttentionProcessor:
    return CARCAttentionProcessor(context_bank, alpha_scheduler, layer_selector)
