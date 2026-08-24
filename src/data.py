import numpy as np
import logging

logger = logging.getLogger(__name__)

class NACReader:
    def __init__(self, path: str):
        self.path = path
        # Hardcoded NAC specs for NAC_POLE_P860N1912.IMG
        self.data_offset = 208988
        self.width = 52247
        self.height = 38443
        self.mm = None

    def __enter__(self):
        self.mm = np.memmap(
            self.path, 
            dtype='<f4', 
            mode='r', 
            offset=self.data_offset, 
            shape=(self.height, self.width)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mm is not None:
            del self.mm

    def extract_patch(self, cx: int, cy: int, size: int) -> np.ndarray:
        half = size // 2
        y1, y2 = max(0, cy - half), min(self.height, cy + half)
        x1, x2 = max(0, cx - half), min(self.width, cx + half)
        
        # Read from memmap
        patch = np.array(self.mm[y1:y2, x1:x2])
        
        # Handle NAC null values
        null_val = np.float32(np.frombuffer(bytes.fromhex('FBFFFF7F'), dtype='<f4')[0])
        patch[patch == null_val] = np.nan
        
        return patch

def simulate_tmc2(nac_patch: np.ndarray, scale_factor: int = 5) -> np.ndarray:
    """Downsample NAC (1m/px) to simulate TMC2 (5m/px) and add transforms."""
    import cv2
    
    # Remove NaNs for resizing by replacing with mean
    valid_mask = np.isfinite(nac_patch)
    if not valid_mask.any():
        return nac_patch
        
    mean_val = np.mean(nac_patch[valid_mask])
    clean_patch = np.where(valid_mask, nac_patch, mean_val)
    
    # Downsample
    h, w = clean_patch.shape
    new_h, new_w = h // scale_factor, w // scale_factor
    tmc_sim = cv2.resize(clean_patch, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Simulate illumination change (simple gamma and contrast shift)
    # Normalize to 0-1 first
    p_min, p_max = np.percentile(tmc_sim, 1), np.percentile(tmc_sim, 99)
    p_max = max(p_max, p_min + 1e-5)
    normalized = np.clip((tmc_sim - p_min) / (p_max - p_min), 0, 1)
    
    # Apply gamma
    gamma = 1.5
    tmc_sim_illum = np.power(normalized, gamma)
    
    # Scale back to 8-bit for feature extractors
    tmc_8bit = (tmc_sim_illum * 255).astype(np.uint8)
    
    # Also prepare the original NAC patch as 8-bit reference
    ref_min, ref_max = np.percentile(clean_patch, 1), np.percentile(clean_patch, 99)
    ref_max = max(ref_max, ref_min + 1e-5)
    ref_norm = np.clip((clean_patch - ref_min) / (ref_max - ref_min), 0, 1)
    ref_8bit = (ref_norm * 255).astype(np.uint8)
    
    return tmc_8bit, ref_8bit
