

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn.functional as F

def _gaussian_kernel(ksize: int, sigma: float, device, dtype):
    """Generates a 2D Gaussian kernel."""
    # Ensure odd ksize
    if ksize % 2 == 0: ksize += 1
    
    ax = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    xx = ax[None, :]
    yy = ax[:, None]
    
    # Correct formula: exp( -(x^2 + y^2) / (2sigma^2) )
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma * sigma))
    
    # Normalize
    kernel = kernel / kernel.sum()
    return kernel[None, None, :, :]  # [1,1,k,k]

def _dilate(m: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Approximate binary dilation using max pooling."""
    pad = k // 2
    return F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)

def _gaussian_blur(m: torch.Tensor, ksize: int = 7, sigma: float = 1.5) -> torch.Tensor:
    kernel = _gaussian_kernel(ksize, sigma, m.device, m.dtype)
    # Reflect padding to avoid dark borders
    pad = ksize // 2
    m_padded = F.pad(m, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(m_padded, kernel)

def _clip_to_box(mask: torch.Tensor, box_xyxy: Tuple[int, int, int, int]) -> torch.Tensor:
    """Zero out mask values outside the bounding box."""
    H, W = mask.shape[-2], mask.shape[-1]
    x1, y1, x2, y2 = box_xyxy
    
    # Create a coordinate grid or just slicing
    # Slicing is faster but we need to keep tensor size
    clipped = torch.zeros_like(mask)
    
    # Python slicing handles bounds gracefully usually, but clamp for safety
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(x2)), min(H, int(y2))
    
    if x2 > x1 and y2 > y1:
        clipped[..., y1:y2, x1:x2] = mask[..., y1:y2, x1:x2]
        
    return clipped

def _make_positional_mask(image_size: Tuple[int, int], position: str, blur: int = 61) -> torch.Tensor:
    """
    Creates a soft positional mask using Sigmoid for smoother transitions.
    """
    H, W = image_size
    # Normalized coordinates [0, 1]
    yy = torch.linspace(0, 1, steps=H).unsqueeze(1).repeat(1, W)
    xx = torch.linspace(0, 1, steps=W).unsqueeze(0).repeat(H, 1)
    
    pos = position.lower()
    sharpness = 10.0  # Controls edge hardness of the initial mask
    
    if "left" in pos:
        # Sigmoid transition at x=0.5
        mask = torch.sigmoid((0.5 - xx) * sharpness)
    elif "right" in pos:
        mask = torch.sigmoid((xx - 0.5) * sharpness)
    elif "top" in pos or "upper" in pos:
        mask = torch.sigmoid((0.5 - yy) * sharpness)
    elif "bottom" in pos or "lower" in pos:
        mask = torch.sigmoid((yy - 0.5) * sharpness)
    elif "center" in pos:
        # Radial gaussian-like
        cx, cy = 0.5, 0.5
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # 1 at center, 0 at r=0.5
        mask = torch.clamp(1.0 - (dist / 0.5), 0, 1)
    else:
        # Full mask
        mask = torch.ones(H, W)

    mask = mask[None, None] # [1,1,H,W]
    
    # Apply Blur for feathering
    # Note: We create tensor on CPU first, caller moves it
    return mask

class DynamicMaskManager:
    """
    Manages masks for Subjects and Background.
    Maintains image-res masks, updates them via Attention, provides latent-res masks.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        latent_size: Tuple[int, int],
        subject_names: List[str],
        init_masks_img: Optional[Dict[str, torch.Tensor]] = None,
        safety_boxes: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
        beta: float = 0.7,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.image_size = image_size
        self.latent_size = latent_size
        self.subject_names = subject_names
        self.beta = float(beta)
        self.device = device or torch.device("cpu")
        self.dtype = dtype or torch.float32

        H, W = image_size
        self.masks_img: Dict[str, torch.Tensor] = {}
        
        # Initialize foreground masks
        for name in subject_names:
            if init_masks_img and name in init_masks_img:
                m = init_masks_img[name].to(self.device, self.dtype)
            else:
                m = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)
            self.masks_img[name] = m

        # Safety boxes
        self.safety_boxes = safety_boxes or {name: (0, 0, W, H) for name in subject_names}
        
        # Compute BG and latent caches
        self._recompute_bg_and_latent()

    def _recompute_bg_and_latent(self):
        """
        Logic: 
        1. BG = 1 - sum(Foregrounds). Clamp to [0,1].
        2. Normalize Foregrounds only where sum > 1 (overlap handling).
        """
        H, W = self.image_size
        
        # 1. Sum of foregrounds
        sum_fg = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)
        for name in self.subject_names:
            sum_fg += self.masks_img[name]
            
        # 2. Handle Overlap: If sum > 1, normalize
        # Mask where overlap occurs
        overlap_mask = sum_fg > 1.0
        if overlap_mask.any():
            # Simple division normalization
            scale = 1.0 / (sum_fg + 1e-6)
            for name in self.subject_names:
                # Only scale where overlap exists
                self.masks_img[name] = torch.where(
                    overlap_mask, 
                    self.masks_img[name] * scale, 
                    self.masks_img[name]
                )
            # Re-sum after norm
            sum_fg = torch.clamp(sum_fg, max=1.0)

        # 3. Compute Background
        self.bg_mask_img = (1.0 - sum_fg).clamp(0.0, 1.0)

        # 4. Resize to Latent Resolution
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
        Evolve masks based on accumulated attention maps.
        """
        H, W = self.image_size
        
        for name, amap in attn_maps_img.items():
            if name not in self.masks_img:
                continue
            
            # Current mask
            m_prev = self.masks_img[name]

            # Process Attention Map
            a = amap.to(self.device, self.dtype)
            
            # 1. Normalize
            if a.max() > 1e-6:
                a = a / a.max()
            
            # 2. Binarize (Threshold)
            # Use dynamic threshold or fixed? Fixed 0.4 is usually safe for attention
            a_bin = (a > 0.4).to(self.dtype)
            
            # 3. Dilate & Blur (Morphology)
            # Expand region slightly to capture contours
            a_dil = _dilate(a_bin, k=9) 
            a_smooth = _gaussian_blur(a_dil, ksize=21, sigma=3.0)
            
            # 4. Momentum Update
            # New = beta * Measured + (1-beta) * Old
            m_new = self.beta * a_smooth + (1.0 - self.beta) * m_prev
            
            # 5. Safety Box Constraint
            box = self.safety_boxes.get(name, (0, 0, W, H))
            m_boxed = _clip_to_box(m_new, box)
            
            self.masks_img[name] = m_boxed.clamp(0.0, 1.0)

        # Re-balance background
        self._recompute_bg_and_latent()

    @staticmethod
    def init_from_positions(
        image_size: Tuple[int, int],
        latent_size: Tuple[int, int],
        subjects: List[Dict[str, str]],
        safety_expand: float = 0.2, # Allow 20% expansion beyond split
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        H, W = image_size
        init_masks = {}
        boxes = {}
        
        for subj in subjects:
            name = subj["name"]
            position = subj.get("position", "center")
            
            # Create Mask
            m = _make_positional_mask(image_size, position)
            if device is not None:
                m = m.to(device, dtype or torch.float32)
            init_masks[name] = m
            
            # Create Safety Box based on position logic
            # Default bounds
            x1, y1, x2, y2 = 0, 0, W, H
            
            if "left" in position:
                x2 = int(0.5 * W)
            elif "right" in position:
                x1 = int(0.5 * W)
            elif "top" in position:
                y2 = int(0.5 * H)
            elif "bottom" in position:
                y1 = int(0.5 * H)
            
            # Expand safety box slightly to allow crossing center
            w_box, h_box = x2 - x1, y2 - y1
            ex = int(w_box * safety_expand)
            ey = int(h_box * safety_expand)
            
            boxes[name] = (
                max(0, x1 - ex), 
                max(0, y1 - ey), 
                min(W, x2 + ex), 
                min(H, y2 + ey)
            )

        return DynamicMaskManager(
            image_size, latent_size, 
            [s["name"] for s in subjects], 
            init_masks, boxes, 
            device=device, dtype=dtype
        )
