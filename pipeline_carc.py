"""
pipeline_carc.py
FINAL V11: With Cross-Attention Gating.
"""
# ... (Imports 和 Helpers 保持不变，省略以节省篇幅) ...
# 请保留原有的 imports, _seed_everything, _prepare_latents, _decode_vae, _attach..., _token...

# 仅修改 CARCPipeline 的 Phase C 部分
class CARCPipeline:
    # ... (__init__, _encode_prompt, _prepare_embeddings, _get_subject_token_ids 不变) ...
    # 为了完整性，建议你只替换 __call__ 方法，或者确保上面 helper 和 init 都在

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
