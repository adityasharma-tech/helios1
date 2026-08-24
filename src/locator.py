import cv2
import numpy as np
import h5py
import logging

def locate_target_region(query_img, index_h5_path):
    """
    Given a query image (numpy array, grayscale) and an h5 index path,
    find the bounding box of the query in the reference image coordinates.
    Returns (cx, cy, size).
    """
    logging.info(f"Loading precomputed reference features from {index_h5_path}")
    with h5py.File(index_h5_path, 'r') as f:
        ref_kps = f['keypoints'][:]
        ref_descs = f['descriptors'][:]
        
    logging.info("Extracting SIFT features from query image...")
    # Ensure query image is 8-bit for SIFT
    if query_img.dtype != np.uint8:
        # Simple normalization if it's not uint8
        query_img = cv2.normalize(query_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    sift = cv2.SIFT_create()
    query_kps, query_descs = sift.detectAndCompute(query_img, None)
    
    if query_descs is None or len(query_descs) == 0:
        raise ValueError("No SIFT features found in the query image.")

    logging.info("Matching features via FLANN...")
    index_params = dict(algorithm=1, trees=5) # KD-Tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(query_descs, ref_descs, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    logging.info(f"Found {len(good_matches)} good SIFT matches.")

    if len(good_matches) < 4:
        raise ValueError("Not enough SIFT matches to localize the query image in the target index.")

    src_pts = np.float32([query_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ref_kps[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

    logging.info("Computing Homography for localization...")
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        raise ValueError("Could not compute valid homography for localization.")
        
    inliers = mask.ravel().tolist().count(1)
    logging.info(f"Localization RANSAC Inliers: {inliers} / {len(good_matches)}")
        
    qh, qw = query_img.shape
    corners = np.float32([[0, 0], [qw, 0], [qw, qh], [0, qh]]).reshape(-1, 1, 2)
    mapped_corners = cv2.perspectiveTransform(corners, H)

    xmin, ymin = np.int32(mapped_corners.min(axis=0).ravel())
    xmax, ymax = np.int32(mapped_corners.max(axis=0).ravel())
    
    # Calculate cx, cy, and size
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    
    w_box = max(1, xmax - xmin)
    h_box = max(1, ymax - ymin)
    
    # Size should be max of width/height, with 20% padding to capture context for Kornia
    base_size = max(w_box, h_box)
    size = int(base_size * 1.2) 
    
    logging.info(f"Localized region: cx={cx}, cy={cy}, size={size} (including padding)")
    return cx, cy, size
