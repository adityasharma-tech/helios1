import os
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_tile(args):
    """
    Worker function to process a single tile.
    Must be defined at the top level for multiprocessing serialization.
    """
    img_path, x, y, w, h = args
    
    # Initialize SIFT inside the worker (it's not easily pickleable)
    sift = cv2.SIFT_create(nfeatures=2000)
    
    with rasterio.open(img_path) as src:
        # Read the specific tile
        tile = src.read(1, window=Window(x, y, w, h))
        
        # Normalize tile to 8-bit for SIFT
        # Using a fixed robust normalization (e.g., 2nd to 98th percentile) can sometimes 
        # be better for lunar images than MINMAX, but MINMAX is fine for general SIFT.
        tile_norm = cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        kps, descs = sift.detectAndCompute(tile_norm, None)
        
        if descs is not None and len(descs) > 0:
            # Convert local tile coordinates to global reference coordinates
            global_kps = np.array([[kp.pt[0] + x, kp.pt[1] + y] for kp in kps], dtype=np.float32)
            return global_kps, descs
            
    return None, None

def build_feature_index_fast(img_path, output_h5="reference_features.h5", tile_size=1024, overlap=128, max_workers=None):
    """
    Parallelized feature extraction pipeline.
    """
    print(f"Analyzing {img_path} for tiling...")
    
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        
    tasks = []
    
    # 1. First, calculate all the tiles we need to process
    with rasterio.open(img_path) as src:
        width, height = src.width, src.height
        step = tile_size - overlap
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                w = min(tile_size, width - x)
                h = min(tile_size, height - y)
                tasks.append((img_path, x, y, w, h))
                
    total_tiles = len(tasks)
    print(f"Generated {total_tiles} tiles. Starting parallel extraction with {max_workers} workers...")

    all_keypoints = []
    all_descriptors = []
    
    # 2. Process tiles in parallel using ProcessPoolExecutor
    processed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_tile, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed}/{total_tiles} tiles...")
                
            kps, descs = future.result()
            
            if kps is not None:
                all_keypoints.append(kps)
                all_descriptors.append(descs)

    # 3. Concatenate and save to HDF5
    if not all_keypoints:
        print("No features were found!")
        return
        
    print("Concatenating features...")
    all_kps_arr = np.vstack(all_keypoints)
    all_descs_arr = np.vstack(all_descriptors)

    print(f"Saving {len(all_kps_arr)} keypoints to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('keypoints', data=all_kps_arr, compression="gzip")
        f.create_dataset('descriptors', data=all_descs_arr, compression="gzip")
        f.attrs['image_width'] = width
        f.attrs['image_height'] = height

    print(f"Success! HDF5 index created at: {output_h5}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build SIFT feature index for a large planetary image.")
    parser.add_argument("image_path", help="Path to the large .img or .tif file")
    parser.add_argument("--output", default="reference_features.h5", help="Output HDF5 file path")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    build_feature_index_fast(
        img_path=args.image_path,
        output_h5=args.output,
        tile_size=args.tile_size,
        max_workers=args.workers
    )
