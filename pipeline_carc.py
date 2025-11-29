"""
pipeline_carc.py
FINAL V10: Fixes Parameter Passing to MaskManager.
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from attention_processor import (
    CARCAttentionProcessor, ContextBank, AlphaScheduler, LayerSelector,
)
from guidance import (
    compute_contrastive_cfg, compute_cfg, prepare_timesteps, _unet_forward
)
from mask_manager import DynamicMaskManager

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
    latents = latents / vae.config.scaling_factor
    latents = latents.to(dtype=vae.dtype)
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image

def _attach_processors_to_unet(unet, processor: CARCAttentionProcessor):
    for n, m in unet.named_modules():
        if hasattr(m, "to_q") and hasattr(m, "to_k"):
            setattr(m, "layer_id", n)
    unet.set_attn_processor(processor)

def _token_indices_for_keywords(tokenizer, prompt: str, keywords: List[str]) -> List[int]:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = [tokenizer.decode([i]).strip().lower() for i in input_ids]
    indices = []
    kw_lower = [k.lower() for k in keywords]
    for i, token_str in enumerate(decoded):
        if any(k in token_str for k in kw_lower):
            indices.append(i) 
    return [i + 1 for i in indices] 

class CARCPipeline:
    def __init__(
        self,
        base_pipe,
        alpha_cfg: Optional[Dict[str, Any]] = None,
        inject_layer_patterns: Optional[List[str]] = None,
    ):
        self.pipe = base_pipe
        self.unet = base_pipe.unet
        self.vae = base_pipe.vae
        self.scheduler = base_pipe.scheduler
        self.device = base_pipe.device
        self.dtype = self.unet.dtype

        self.context_bank = ContextBank()
        alpha_scheduler = AlphaScheduler(**(alpha_cfg or {}))
        layer_selector = LayerSelector(inject_layer_patterns)
        
        self.attn_proc = CARCAttentionProcessor(self.context_bank, alpha_scheduler, layer_selector)
        _attach_processors_to_unet(self.unet, self.attn_proc)

    def _encode_prompt(self, prompt: str, negative_prompt: str = ""):
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
        embs = {}
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=torch.float32, device=self.device)
        
        def pack_cond(pos_emb, neg_emb, pos_pool, neg_pool):
            return {
                "pos": pos_emb,
                "uncond": neg_emb,
                "added_pos": {"text_embeds": pos_pool, "time_ids": add_time_ids},
                "added_uncond": {"text_embeds": neg_pool, "time_ids": add_time_ids},
            }

        pe, ne, pp, np = self._encode_prompt(global_prompt, "")
        embs["global"] = pack_cond(pe, ne, pp, np)
        
        for subj in subjects:
            name = subj["name"]
            spe, sne, spp, snp = self._encode_prompt(subj["prompt"], "")
            neg_target_text = subj.get("neg_target", "")
            if neg_target_text:
                nt_pe, _, nt_pp, _ = self._encode_prompt(neg_target_text, "")
                embs[name] = pack_cond(spe, sne, spp, snp)
                embs[name]["neg_target"] = nt_pe
                embs[name]["added_neg_target"] = {"text_embeds": nt_pp, "time_ids": add_time_ids}
            else:
                embs[name] = pack_cond(spe, sne, spp, snp)
                embs[name]["neg_target"] = None
        return embs

    def _get_subject_token_ids(self, subjects):
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
        bg_update_interval: int = 2,
        beta: float = 0.6,
        safety_expand: float = 0.05,
        bg_floor: float = 0.05,
        gap_ratio: float = 0.06,
        kappa: float = 1.5,
        seed: int = 42,
    ):
        _seed_everything(seed)
        
        # ... (Setup 和 Phase A/B 不变) ...
        # 这里简写，请确保你保留了之前的 Setup 代码
        embs = self._prepare_embeddings(global_prompt, subjects, width, height)
        self.attn_proc.set_subject_token_ids(self._get_subject_token_ids(subjects))
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        latent_h, latent_w = height // 8, width // 8
        latents = _prepare_latents(1, 4, latent_h, latent_w, self.dtype, self.device, self.scheduler, generator=torch.Generator(device=self.device).manual_seed(seed))
        mask_mgr = DynamicMaskManager.init_from_positions((height, width), (latent_h, latent_w), subjects, safety_expand=safety_expand, bg_floor=bg_floor, gap_ratio=gap_ratio, device=self.device, dtype=self.dtype)
        mask_mgr.beta = beta
        self.attn_proc.set_base_latent_hw(latent_h, latent_w)
        
        for i, t in enumerate(timesteps):
            self.attn_proc.set_step_index(i, num_inference_steps)
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            
            # Phase A: Background
            if i % bg_update_interval == 0:
                self.context_bank.clear()
                self.attn_proc.set_mode("record_bg")
                self.attn_proc.set_probe_enabled(False)
                # 清除区域掩码，背景不需要门控
                self.attn_proc.set_region_mask(None) 
                _unet_forward(self.unet, latent_model_input, t, embs["global"]["pos"], embs["global"]["added_pos"])
            
            # Phase B
            self.attn_proc.set_mode("off")
            self.attn_proc.set_region_mask(None)
            eps_bg = compute_cfg(self.unet, latent_model_input, t, embs["global"]["pos"], embs["global"]["uncond"], embs["global"]["added_pos"], embs["global"]["added_uncond"], cfg_pos)
            
            # Phase C: Subjects
            do_probe = (i % mask_update_interval == 0)
            eps_subjects = {}
            
            # 获取当前掩码用于门控
            masks_latent_gate, _ = mask_mgr.get_masks_latent()
            
            for sid, subj in enumerate(subjects):
                name = subj["name"]
                self.attn_proc.set_subject_id(sid)
                s_emb = embs[name]
                
                # 【新增】设置区域门控掩码！
                # 这会传给 Attention Processor，在 Cross-Attn 时把 mask 外的权重置零
                self.attn_proc.set_region_mask(masks_latent_gate[name])
                
                # C.1 Probe
                if do_probe:
                    self.attn_proc.set_mode("inject_subject")
                    self.attn_proc.set_probe_enabled(True)
                    _unet_forward(self.unet, latent_model_input, t, s_emb["pos"], s_emb["added_pos"])
                
                # C.2 Noise
                self.attn_proc.set_mode("inject_subject")
                self.attn_proc.set_probe_enabled(False)
                
                if s_emb["neg_target"] is not None:
                    eps_sub = compute_contrastive_cfg(self.unet, latent_model_input, t, s_emb["pos"], s_emb["uncond"], s_emb["neg_target"], s_emb["added_pos"], s_emb["added_uncond"], cfg_pos, cfg_neg)
                else:
                    eps_sub = compute_cfg(self.unet, latent_model_input, t, s_emb["pos"], s_emb["uncond"], s_emb["added_pos"], s_emb["added_uncond"], cfg_pos)
                eps_subjects[name] = eps_sub
                
            # 【重要】跑完主体后清除掩码，以免影响后续步骤（虽然下一轮循环开始会重置，但清空是好习惯）
            self.attn_proc.set_region_mask(None)

            # Phase D
            masks_latent, bg_mask_latent = mask_mgr.get_masks_latent()
            def expand(m): return m.expand(latents.shape[0], 4, -1, -1)
            eps_final = eps_bg 
            for name, eps_s in eps_subjects.items():
                w = expand(masks_latent[name])
                eps_final = eps_final + kappa * w * (eps_s - eps_bg)
            latents = self.scheduler.step(eps_final, t, latents).prev_sample
            
            # Phase E
            if do_probe:
                attn_maps = self.attn_proc.pop_attn_maps()
                maps_img = {}
                for sid, amap in attn_maps.items():
                    name = subjects[sid]["name"]
                    maps_img[name] = F.interpolate(amap, size=(height, width), mode="bilinear", align_corners=False)
                if maps_img: mask_mgr.update_from_attn(maps_img)

        image = _decode_vae(self.vae, latents)
        return {"image": image}

def save_output(out, path):
    save_image(out["image"], path)
