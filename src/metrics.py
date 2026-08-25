import numpy as np

def calculate_rmse(pts1: np.ndarray, pts2: np.ndarray, M: np.ndarray, inliers: np.ndarray):
    """Calculate RMSE of the transformed points vs target points."""
    if not inliers.any():
        return float('inf')
        
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    
    # Apply homography to pts1
    import cv2
    pts1_trans = cv2.perspectiveTransform(pts1_in.reshape(-1, 1, 2), M).reshape(-1, 2)
    
    # Calculate Euclidean distance
    diff = pts1_trans - pts2_in
    sq_dist = np.sum(diff**2, axis=1)
    
    return np.sqrt(np.mean(sq_dist))

def calculate_psnr(img1: np.ndarray, img2: np.ndarray):
    """Peak Signal-to-Noise Ratio (PSNR)."""
    import cv2
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1: np.ndarray, img2: np.ndarray):
    """Structural Similarity Index Measure (SSIM)."""
    # Simplified SSIM implementation using cv2
    import cv2
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def check_uniformity(pts: np.ndarray, img_shape: tuple, grid_size: tuple = (4, 4)):
    """Check if matches are uniformly distributed across the image."""
    h, w = img_shape[:2]
    gh, gw = grid_size
    
    grid = np.zeros(grid_size, dtype=int)
    for pt in pts:
        x, y = pt
        gx = min(int((x / w) * gw), gw - 1)
        gy = min(int((y / h) * gh), gh - 1)
        grid[gy, gx] += 1
        
    occupied_cells = np.sum(grid > 0)
    total_cells = gh * gw
    
    return occupied_cells / total_cells
