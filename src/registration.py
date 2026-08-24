import cv2
import torch
import kornia.feature as KF
import kornia
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureMatcher:
    def __init__(self, method: str = 'disk'):
        self.method = method.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.method == 'disk':
            logger.info(f"Loading DISK + LightGlue on {self.device}")
            self.extractor = KF.DISK.from_pretrained("depth").to(self.device).eval()
            self.matcher = KF.LightGlue("disk").to(self.device).eval()
        elif self.method == 'sift':
            logger.info("Loading SIFT (CPU fallback)")
            self.sift = cv2.SIFT_create(nfeatures=10000)
            # FLANN matcher for SIFT
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unknown method {self.method}")

    def extract_and_match(self, img1: np.ndarray, img2: np.ndarray):
        """Extract features and match them between img1 and img2."""
        if self.method == 'disk':
            return self._match_disk(img1, img2)
        else:
            return self._match_sift(img1, img2)

    def _match_disk(self, img1: np.ndarray, img2: np.ndarray):
        # Convert to torch tensor
        def to_tensor(img):
            t = kornia.image_to_tensor(img, keepdim=False).to(self.device)
            t = t.float() / 255.0
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
            return t

        t1, t2 = to_tensor(img1), to_tensor(img2)
        
        with torch.inference_mode():
            # Extract features
            f1 = self.extractor(t1, n=5000, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
            f2 = self.extractor(t2, n=5000, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
            
            # Match features
            lafs1 = kornia.feature.laf_from_center_scale_ori(
                f1.keypoints[None], torch.ones(1, len(f1.keypoints), 1, 1, device=self.device)
            )
            lafs2 = kornia.feature.laf_from_center_scale_ori(
                f2.keypoints[None], torch.ones(1, len(f2.keypoints), 1, 1, device=self.device)
            )
            
            dists, idxs = self.matcher(f1.descriptors, f2.descriptors, lafs1, lafs2)
            
        pts1 = f1.keypoints[idxs[:, 0]].cpu().numpy()
        pts2 = f2.keypoints[idxs[:, 1]].cpu().numpy()
        
        # Convert to cv2 KeyPoints and DMatch for compatibility
        kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in f1.keypoints.cpu().numpy()]
        kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in f2.keypoints.cpu().numpy()]
        matches = [cv2.DMatch(_queryIdx=int(i), _trainIdx=int(j), _distance=0) for i, j in idxs.cpu().numpy()]
        
        return pts1, pts2, kp1, kp2, matches

    def _match_sift(self, img1: np.ndarray, img2: np.ndarray):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or len(des1) < 2 or des2 is None or len(des2) < 2:
            return np.empty((0,2)), np.empty((0,2)), kp1, kp2, []
            
        matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good = []
        pts1, pts2 = [], []
        for match_group in matches:
            if len(match_group) == 2:
                m, n = match_group
                if m.distance < 0.75 * n.distance:
                    good.append(m)
                    pts1.append(kp1[m.queryIdx].pt)
                    pts2.append(kp2[m.trainIdx].pt)
                    
        return np.array(pts1), np.array(pts2), kp1, kp2, good

def geometric_verification(pts1: np.ndarray, pts2: np.ndarray):
    """Run RANSAC to find robust transformation and inliers."""
    if len(pts1) < 4:
        return None, np.array([])
        
    M, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 5.0)
    if M is None or mask is None:
        return None, np.array([])
        
    inliers = mask.ravel() == 1
    return M, inliers

def refine_subpixel(img1: np.ndarray, img2: np.ndarray, M: np.ndarray):
    """Phase correlation for subpixel refinement."""
    h, w = img2.shape
    warped = cv2.warpPerspective(img1, M, (w, h))
    
    w_f32 = np.float32(warped)
    i_f32 = np.float32(img2)
    
    shift, response = cv2.phaseCorrelate(w_f32, i_f32)
    return shift, response
