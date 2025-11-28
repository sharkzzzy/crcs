"""
core/mask_manager.py
FINAL V8: Erosion-based Mask Update & Lower BG Floor.
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn.functional as F

def _gaussian_kernel(ksize: int, sigma: float, device, dtype):
    if ksize % 2 == 0: ksize += 1
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    xx = ax[None, :]
    yy = ax[:, None]
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel[None, None, :, :]

def _dilate(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    return F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)

def _gaussian_blur(m: torch.Tensor, ksize: int = 7, sigma: float = 1.5) -> torch.Tensor:
    kernel = _gaussian_kernel(ksize, sigma, m.device, m.dtype)
    pad = ksize // 2
    m_padded = F.pad(m, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(m_padded, kernel)

def _clip_to_box(mask: torch.Tensor, box_xyxy: Tuple[int, int, int, int]) -> torch.Tensor:
    H, W = mask.shape[-2], mask.shape[-1]
    x1, y1, x2, y2 = box_xyxy
    clipped = torch.zeros_like(mask)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    if x2 > x1 and y2 > y1:
        clipped[..., y1:y2, x1:x2] = mask[..., y1:y2, x1:x2]
    return clipped

def _make_positional_mask(image_size: Tuple[int, int], position: str, sharpness: float = 6.0) -> torch.Tensor:
    H, W = image_size
    yy = torch.linspace(0, 1, steps=H).unsqueeze(1).repeat(1, W)
    xx = torch.linspace(0, 1, steps=W).unsqueeze(0).repeat(H, 1)
    pos = position.lower()
    if "left" in pos: mask = torch.sigmoid((0.5 - xx) * sharpness)
    elif "right" in pos: mask = torch.sigmoid((xx - 0.5) * sharpness)
    elif "top" in pos: mask = torch.sigmoid((0.5 - yy) * sharpness)
    elif "bottom" in pos: mask = torch.sigmoid((yy - 0.5) * sharpness)
    else: mask = torch.ones(H, W)
    return mask[None, None]

class DynamicMaskManager:
    def __init__(
        self,
        image_size: Tuple[int, int],
        latent_size: Tuple[int, int],
        subject_names: List[str],
        init_masks_img: Optional[Dict[str, torch.Tensor]] = None,
        safety_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        beta: float = 0.6, # 降低动量
        bg_floor: float = 0.05,  # 【改动】降低背景底权重到 0.05
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.image_size = image_size
        self.latent_size = latent_size
        self.subject_names = subject_names
        self.beta = float(beta)
        self.bg_floor = float(bg_floor)
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

        H, W = image_size
        self.masks_img: Dict[str, torch.Tensor] = {}
        for name in subject_names:
            if init_masks_img and name in init_masks_img:
                m = init_masks_img[name].to(self.device, self.dtype)
            else:
                m = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)
            self.masks_img[name] = m

        self.safety_boxes = safety_boxes or {name: (0, 0, W, H) for name in subject_names}
        self._recompute_bg_and_latent()

    def _recompute_bg_and_latent(self):
        H, W = self.image_size
        sum_fg = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)
        for name in self.subject_names:
            sum_fg = sum_fg + self.masks_img[name]
            
        w_bg = (1.0 - sum_fg).clamp(0.0, 1.0) + self.bg_floor
        denom = w_bg + sum_fg + 1e-6
        
        self.bg_mask_img = w_bg / denom
        for name in self.subject_names:
            self.masks_img[name] = self.masks_img[name] / denom

        H_lat, W_lat = self.latent_size
        self.masks_latent = {}
        for name in self.subject_names:
            self.masks_latent[name] = F.interpolate(
                self.masks_img[name], size=(H_lat, W_lat), mode="bilinear", align_corners=False
            )
        self.bg_mask_latent = F.interpolate(
            self.bg_mask_img, size=(H_lat, W_lat), mode="bilinear", align_corners=False
        )

    def get_masks_latent(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.masks_latent, self.bg_mask_latent

    def update_from_attn(self, attn_maps_img: Dict[str, torch.Tensor]):
        """
        【改动】使用腐蚀替代膨胀，提高阈值。
        """
        H, W = self.image_size
        for name, amap in attn_maps_img.items():
            if name not in self.masks_img: continue
            
            m_prev = self.masks_img[name]
            a = amap.to(self.device, self.dtype)
            if a.max() > 1e-6: a = a / a.max()
            
            # 提高阈值到 0.55
            a_bin = (a > 0.55).to(self.dtype)
            
            # 【改动】腐蚀：1 - dilate(1 - mask)
            # 先反转背景，膨胀背景 = 腐蚀前景
            a_bg = 1.0 - a_bin
            a_bg_dilated = _dilate(a_bg, k=5)
            a_erode = 1.0 - a_bg_dilated
            
            # 平滑
            a_smooth = _gaussian_blur(a_erode, ksize=15, sigma=2.5)
            
            m_new = self.beta * a_smooth + (1.0 - self.beta) * m_prev
            box = self.safety_boxes.get(name, (0, 0, W, H))
            m_boxed = _clip_to_box(m_new, box)
            
            self.masks_img[name] = m_boxed.clamp(0.0, 1.0)

        self._recompute_bg_and_latent()

    @staticmethod
    def init_from_positions(
        image_size: Tuple[int, int], latent_size: Tuple[int, int], subjects: List[Dict[str, str]],
        safety_expand: float = 0.2, bg_floor: float = 0.05, gap_ratio: float = 0.05,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    ):
        H, W = image_size
        init_masks = {}
        boxes = {}
        gap_w = max(2, int(W * gap_ratio))
        center_l, center_r = W // 2 - gap_w // 2, W // 2 + gap_w // 2
        
        for subj in subjects:
            name = subj["name"]
            position = subj.get("position", "center")
            m = _make_positional_mask(image_size, position, sharpness=6.0)
            if device: m = m.to(device, dtype or torch.float32)
            if "left" in position or "right" in position:
                m[..., :, center_l:center_r] = 0.0
            m = _gaussian_blur(m, ksize=31, sigma=6.0)
            init_masks[name] = m
            
            x1, y1, x2, y2 = 0, 0, W, H
            if "left" in position: x2 = int(0.5 * W)
            elif "right" in position: x1 = int(0.5 * W)
            
            w_box, h_box = x2 - x1, y2 - y1
            ex, ey = int(w_box * safety_expand), int(h_box * safety_expand)
            boxes[name] = (max(0, x1 - ex), max(0, y1 - ey), min(W, x2 + ex), min(H, y2 + ey))

        return DynamicMaskManager(image_size, latent_size, [s["name"] for s in subjects], init_masks, boxes, beta=0.6, bg_floor=bg_floor, device=device, dtype=dtype)
