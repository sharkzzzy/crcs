"""
core/guidance.py
FINAL V2: Sequential Forward for Peak Memory Reduction.
"""

from typing import Optional, Dict, Any, Tuple
import torch

def _unet_forward(
    unet, 
    latents: torch.Tensor, 
    t: torch.Tensor, 
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
) -> torch.Tensor:
    """Helper for single forward pass."""
    out = unet(
        latents, 
        t, 
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs
    )
    if isinstance(out, dict):
        return out.get("sample", out.get("out"))
    return getattr(out, "sample", out)

@torch.no_grad()
def compute_cfg(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    emb_pos: torch.Tensor,
    emb_uncond: torch.Tensor,
    added_cond_kwargs_pos: Dict[str, torch.Tensor],
    added_cond_kwargs_uncond: Dict[str, torch.Tensor],
    cfg_pos: float,
) -> torch.Tensor:
    """
    Standard CFG: Sequential Execution to save VRAM.
    """
    # 1. Uncond Pass
    eps_uncond = _unet_forward(unet, latents, t, emb_uncond, added_cond_kwargs_uncond)
    
    # 2. Pos Pass
    eps_pos = _unet_forward(unet, latents, t, emb_pos, added_cond_kwargs_pos)
    
    # Combine
    return eps_uncond + cfg_pos * (eps_pos - eps_uncond)

@torch.no_grad()
def compute_contrastive_cfg(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    emb_pos: torch.Tensor,
    emb_uncond: torch.Tensor,
    emb_neg_target: torch.Tensor,
    added_cond_kwargs_pos: Dict[str, torch.Tensor],
    added_cond_kwargs_uncond: Dict[str, torch.Tensor],
    cfg_pos: float,
    cfg_neg: float,
) -> torch.Tensor:
    """
    Contrastive Guidance: Sequential Execution.
    eps = uncond + s1*(pos-uncond) - s2*(neg-uncond)
    """
    # 1. Uncond Pass
    eps_uncond = _unet_forward(unet, latents, t, emb_uncond, added_cond_kwargs_uncond)
    
    # 2. Pos Pass
    eps_pos = _unet_forward(unet, latents, t, emb_pos, added_cond_kwargs_pos)
    
    # 3. Neg Target Pass (Neg Target usually reuses uncond structure info)
    eps_neg = _unet_forward(unet, latents, t, emb_neg_target, added_cond_kwargs_uncond)
    
    # Combine
    return eps_uncond + cfg_pos * (eps_pos - eps_uncond) - cfg_neg * (eps_neg - eps_uncond)

def prepare_timesteps(scheduler, num_inference_steps: int, device: torch.device):
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
    return timesteps
