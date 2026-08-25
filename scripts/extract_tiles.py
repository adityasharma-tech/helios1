import os
import glob
import random
import cv2
import numpy as np

# Add project root to path so we can import src modules
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data import ISROReader, normalize_16bit_to_8bit

def main():
    source_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO/ch2_ohr_ncp_202/zip/data/calibrated"
    output_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO/isro_0111"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .img files recursively
    img_files = glob.glob(os.path.join(source_dir, "**", "*.img"), recursive=True)
    
    if not img_files:
        print(f"No .img files found in {source_dir}")
        return
        
    print(f"Found {len(img_files)} .img files. Beginning extraction of 1000 patches...")
    
    total_images_needed = 1000
    images_per_file = total_images_needed // len(img_files)
    remainder = total_images_needed % len(img_files)
    
    extracted_count = 0
    
    for i, img_path in enumerate(img_files):
        xml_path = img_path.replace(".img", ".xml")
        if not os.path.exists(xml_path):
            print(f"Warning: XML missing for {img_path}, skipping.")
            continue
            
        print(f"Processing {os.path.basename(img_path)}...")
        
        target_count = images_per_file + (1 if i < remainder else 0)
        
        try:
            with ISROReader(img_path, xml_path) as reader:
                h, w = reader.height, reader.width
                print(f"  -> Dimensions: {w}x{h}, extracting {target_count} patches")
                
                for j in range(target_count):
                    # Pick a random size for the patch (e.g. between 600 and 1500 pixels)
                    size = random.randint(600, 1500)
                    
                    # Ensure coordinates are within valid bounds
                    if w <= size or h <= size:
                        continue
                        
                    cx = random.randint(size // 2, w - (size // 2) - 1)
                    cy = random.randint(size // 2, h - (size // 2) - 1)
                    
                    patch = reader.extract_patch(cx, cy, size)
                    
                    # Convert to 8-bit for standard PNG saving
                    patch_8bit = normalize_16bit_to_8bit(patch)
                    
                    # Generate a unique filename using timestamp and coordinates
                    file_id = os.path.basename(img_path).split('_')[3]
                    out_filename = os.path.join(output_dir, f"patch_{file_id}_{cx}_{cy}_{size}.png")
                    
                    cv2.imwrite(out_filename, patch_8bit)
                    extracted_count += 1
                    
                    if extracted_count % 100 == 0:
                        print(f"[{extracted_count}/{total_images_needed}] patches extracted...")
                        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    print(f"\nDone! Successfully extracted {extracted_count} images to {output_dir}")

if __name__ == "__main__":
    main()
