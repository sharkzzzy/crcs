from typing import Optional, Dict, Any, Tuple
import torch

def _unet_forward(
    unet, 
    latents: torch.Tensor, 
    t: torch.Tensor, 
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Unified UNet forward helper. 
    Crucial Fix: Passes added_cond_kwargs for SDXL.
    """
    out = unet(
        latents, 
        t, 
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs
    )
    
    if isinstance(out, dict):
        eps = out.get("sample", None)
        if eps is None and "out" in out:
            eps = out["out"]
    else:
        eps = getattr(out, "sample", out)
        
    return eps


@torch.no_grad()
def compute_cfg(
    unet,
    latents: torch.Tensor,
    t: torch.Tensor,
    emb_pos: torch.Tensor,
    emb_uncond: torch.Tensor,
    added_cond_kwargs_pos: Dict[str, torch.Tensor],    # SDXL specific
    added_cond_kwargs_uncond: Dict[str, torch.Tensor], # SDXL specific
    cfg_pos: float,
) -> torch.Tensor:
    """
    Standard CFG: eps = eps_uncond + cfg * (eps_pos - eps_uncond)
    """
    # Batching: [Uncond, Pos]
    lat_batched = latents.repeat(2, 1, 1, 1)
    t_batched = t.repeat(2)
    emb_batched = torch.cat([emb_uncond, emb_pos], dim=0)
    
    # Batch added_cond_kwargs (text_embeds, time_ids)
    # They are tensors, so we cat them along dim 0
    added_cond_batched = {}
    for k in added_cond_kwargs_pos.keys():
        v_pos = added_cond_kwargs_pos[k]
        v_uncond = added_cond_kwargs_uncond[k]
        added_cond_batched[k] = torch.cat([v_uncond, v_pos], dim=0)

    # Forward
    out = _unet_forward(unet, lat_batched, t_batched, emb_batched, added_cond_batched)
    eps_uncond, eps_pos = out.chunk(2, dim=0)
    
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
    # Assume neg_target uses same time_ids/pooled as pos or uncond? 
    # Usually safer to use uncond's pooled/time_ids for neg_target.
    cfg_pos: float,
    cfg_neg: float,
) -> torch.Tensor:
    """
    Contrastive guidance: 
    eps = eps_uncond + s_pos*(pos - uncond) - s_neg*(neg_target - uncond)
    """
    # Batching: [Uncond, Pos, NegTarget]
    lat_batched = latents.repeat(3, 1, 1, 1)
    t_batched = t.repeat(3)
    emb_batched = torch.cat([emb_uncond, emb_pos, emb_neg_target], dim=0)
    
    # Prepare added kwargs
    # Neg target typically reuses uncond's structural conditions (pooled text etc)
    # unless specific neg embedding is provided. 
    # Here we assume neg_target reuses uncond's added_cond (standard practice).
    added_cond_batched = {}
    for k in added_cond_kwargs_pos.keys():
        v_pos = added_cond_kwargs_pos[k]
        v_uncond = added_cond_kwargs_uncond[k]
        # order: [uncond, pos, neg_target] -> [v_uncond, v_pos, v_uncond]
        added_cond_batched[k] = torch.cat([v_uncond, v_pos, v_uncond], dim=0)

    # Forward
    out = _unet_forward(unet, lat_batched, t_batched, emb_batched, added_cond_batched)
    eps_uncond, eps_pos, eps_neg = out.chunk(3, dim=0)
    
    # Formula: push towards pos, push away from neg
    # Note: Using sub (minus) for neg guidance
    eps = eps_uncond + cfg_pos * (eps_pos - eps_uncond) - cfg_neg * (eps_neg - eps_uncond)
    
    return eps


def prepare_timesteps(scheduler, num_inference_steps: int, device: torch.device):
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps, device=device, dtype=torch.long)
    return timesteps
