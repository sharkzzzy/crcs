

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torchvision.utils import save_image 

from .attention_processor import (
    CARCAttentionProcessor, ContextBank, AlphaScheduler, LayerSelector,
)
from .guidance import compute_contrastive_cfg, compute_cfg, prepare_timesteps
from .mask_manager import DynamicMaskManager

def _seed_everything(seed: int):
    import random, numpy as np
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def _prepare_latents(batch_size, channels, height, width, dtype, device, scheduler, generator=None):
    shape = (batch_size, channels, height, width)
    latents = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    latents = latents * scheduler.init_noise_sigma
    return latents

def _decode_vae(vae, latents: torch.Tensor) -> torch.Tensor:
    # Scale back
    latents = latents / vae.config.scaling_factor
    # Decode
    with torch.no_grad():
        image = vae.decode(latents).sample
    # Normalize to [0,1]
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def _attach_processors_to_unet(unet, processor: CARCAttentionProcessor):
    """
    Attach processor and label layers with 'layer_id'.
    """
    # 1. Label layers first
    for n, m in unet.named_modules():
        if hasattr(m, "to_q") and hasattr(m, "to_k"):
            setattr(m, "layer_id", n)
            
    # 2. Attach processor
    unet.set_attn_processor(processor)

def _token_indices_for_keywords(tokenizer, prompt: str, keywords: List[str]) -> List[int]:
    """Simple heuristic to find tokens matching keywords."""
    # This is a simplified version. For robustness, better use tokenizer encode.
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = [tokenizer.decode([i]).strip().lower() for i in input_ids]
    
    indices = []
    kw_lower = [k.lower() for k in keywords]
    
    for i, token_str in enumerate(decoded):
        if any(k in token_str for k in kw_lower):
            indices.append(i) # Note: indices relative to encoded sequence (excluding start token if stripped)
    
    # Adjust for special tokens usually added by pipeline (start token)
    # SDXL tokenizer usually adds 1 special token at start
    return [i + 1 for i in indices] 


class CARCPipeline:
    """
    Main entry point for CARC generation.
    Pass the original SDXL pipeline to reuse its components and logic.
    """
    def __init__(
        self,
        base_pipe, # Pass StableDiffusionXLPipeline here
        alpha_cfg: Optional[Dict[str, Any]] = None,
        inject_layer_patterns: Optional[List[str]] = None,
    ):
        self.pipe = base_pipe
        self.unet = base_pipe.unet
        self.vae = base_pipe.vae
        self.scheduler = base_pipe.scheduler
        self.device = base_pipe.device
        self.dtype = self.unet.dtype

        # Init Processor
        self.context_bank = ContextBank()
        alpha_scheduler = AlphaScheduler(**(alpha_cfg or {}))
        layer_selector = LayerSelector(inject_layer_patterns)
        
        self.attn_proc = CARCAttentionProcessor(self.context_bank, alpha_scheduler, layer_selector)
        _attach_processors_to_unet(self.unet, self.attn_proc)

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
        """
        Reuse base pipe's complex encode logic to get:
        - prompt_embeds, negative_prompt_embeds
        - pooled_prompt_embeds, negative_pooled_prompt_embeds
        """
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        return (
            prompt_embeds, negative_prompt_embeds, 
            pooled_prompt_embeds, negative_pooled_prompt_embeds
        )

    def _prepare_embeddings(self, global_prompt, subjects, width, height):
        """
        Prepares a dictionary containing both text embeds AND added_cond_kwargs (time_ids).
        """
        embs = {}
        
        # 1. Shared Time IDs (Resolution)
        # SDXL needs original_size, crops_coords_top_left, target_size
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        
        # Create standard time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=self.dtype, device=self.device)
        
        def pack_cond(pos_emb, neg_emb, pos_pool, neg_pool):
            return {
                "pos": pos_emb,
                "uncond": neg_emb,
                "added_pos": {"text_embeds": pos_pool, "time_ids": add_time_ids},
                "added_uncond": {"text_embeds": neg_pool, "time_ids": add_time_ids},
            }

        # Global
        pe, ne, pp, np = self._encode_prompt(global_prompt, "")
        embs["global"] = pack_cond(pe, ne, pp, np)
        
        # Subjects
        for subj in subjects:
            name = subj["name"]
            # Encode Subject Positive
            spe, sne, spp, snp = self._encode_prompt(subj["prompt"], "")
            
            # Encode Negative Target (e.g. "a blue dog") if provided
            neg_target_text = subj.get("neg_target", "")
            if neg_target_text:
                nt_pe, _, nt_pp, _ = self._encode_prompt(neg_target_text, "") # We only need the positive embed of the negative target
                # We use the negative target as the "negative" in contrastive loss? 
                # Actually, in contrastive guidance: eps = eps_uncond + s*(eps_pos - eps_uncond) - s_neg*(eps_neg_target - eps_uncond)
                # So we need "neg_target" as a distinct condition.
                
                # We reuse the uncond (empty) embeddings from the subject for the base
                embs[name] = pack_cond(spe, sne, spp, snp)
                embs[name]["neg_target"] = nt_pe
                # Reuse structural condition from pos or uncond for neg_target (safe bet)
                embs[name]["added_neg_target"] = {"text_embeds": nt_pp, "time_ids": add_time_ids}
            else:
                embs[name] = pack_cond(spe, sne, spp, snp)
                embs[name]["neg_target"] = None
                
        return embs

    def _get_subject_token_ids(self, subjects):
        # Using pipe.tokenizer (CLIP ViT-L)
        mapping = {}
        for i, subj in enumerate(subjects):
            mapping[i] = _token_indices_for_keywords(
                self.pipe.tokenizer, subj["prompt"], [subj["name"]]
            )
        return mapping

    @torch.no_grad()
    def __call__(
        self,
        global_prompt: str,
        subjects: List[Dict[str, str]],
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        cfg_pos: float = 7.5,
        cfg_neg: float = 3.0,
        mask_update_interval: int = 5,
        beta: float = 0.7,
        safety_expand: float = 0.15,
        seed: int = 42,
    ):
        _seed_everything(seed)
        
        # 1. Embeddings
        embs = self._prepare_embeddings(global_prompt, subjects, width, height)
        
        # 2. Token Indices
        self.attn_proc.set_subject_token_ids(self._get_subject_token_ids(subjects))
        
        # 3. Latents
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        latents = _prepare_latents(
            1, 4, height, width, self.dtype, self.device, self.scheduler,
            generator=torch.Generator(device=self.device).manual_seed(seed)
        )
        
        # 4. Masks
        latent_h, latent_w = height // 8, width // 8
        mask_mgr = DynamicMaskManager.init_from_positions(
            (height, width), (latent_h, latent_w), subjects, safety_expand, 
            device=self.device, dtype=self.dtype
        )

        # 5. Loop
        self.attn_proc.set_base_latent_hw(latent_h, latent_w)
        
        for i, t in enumerate(timesteps):
            self.attn_proc.set_step(int(t), num_inference_steps)
            
            # --- A. Background (Record) ---
            self.context_bank.clear()
            self.attn_proc.set_mode("record_bg")
            self.attn_proc.set_probe_enabled(False)
            
            g_emb = embs["global"]
            eps_bg = compute_cfg(
                self.unet, latents, t, 
                g_emb["pos"], g_emb["uncond"],
                g_emb["added_pos"], g_emb["added_uncond"],
                cfg_pos
            )
            
            # --- B. Subjects (Inject + Contrastive) ---
            probe_now = (i % mask_update_interval == 0)
            self.attn_proc.set_mode("inject_subject")
            self.attn_proc.set_probe_enabled(probe_now)
            
            eps_subjects = {}
            for sid, subj in enumerate(subjects):
                name = subj["name"]
                self.attn_proc.set_subject_id(sid)
                
                s_emb = embs[name]
                if s_emb["neg_target"] is not None:
                    # Contrastive
                    eps_sub = compute_contrastive_cfg(
                        self.unet, latents, t,
                        s_emb["pos"], s_emb["uncond"], s_emb["neg_target"],
                        s_emb["added_pos"], s_emb["added_uncond"], 
                        # Assuming neg_target reuses added_pos/uncond structure
                        cfg_pos, cfg_neg
                    )
                else:
                    # Standard
                    eps_sub = compute_cfg(
                        self.unet, latents, t,
                        s_emb["pos"], s_emb["uncond"],
                        s_emb["added_pos"], s_emb["added_uncond"],
                        cfg_pos
                    )
                eps_subjects[name] = eps_sub

            # --- C. Compose ---
            masks_latent, bg_mask_latent = mask_mgr.get_masks_latent()
            
            # Expand masks to [B, 4, H, W]
            def expand(m): return m.expand(latents.shape[0], 4, -1, -1)
            
            eps_final = eps_bg * expand(bg_mask_latent)
            for name, eps_s in eps_subjects.items():
                eps_final += eps_s * expand(masks_latent[name])
                
            # --- D. Step ---
            latents = self.scheduler.step(eps_final, t, latents).prev_sample
            
            # Update Masks
            if probe_now:
                attn_maps = self.attn_proc.pop_attn_maps()
                # Upsample latent maps to image res for manager
                maps_img = {}
                for sid, amap in attn_maps.items():
                    name = subjects[sid]["name"]
                    maps_img[name] = F.interpolate(amap, size=(height, width), mode="bilinear")
                
                if maps_img:
                    mask_mgr.update_from_attn(maps_img)

        # 6. Decode
        image = _decode_vae(self.vae, latents)
        return {"image": image}

def save_output(out, path):
    save_image(out["image"], path)
