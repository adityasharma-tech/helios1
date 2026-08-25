import os
import cv2
import numpy as np
from src.data import ISROReader, normalize_16bit_to_8bit

def split_isro_image():
    isro_img_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.img"
    isro_xml_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"
    
    # Create a folder for square tiles
    output_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO/mosaics/100_squares"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Processing ISRO OHRC Image to create square tiles...")
    with ISROReader(isro_img_path, isro_xml_path) as isro_reader:
        isro_h, isro_w = isro_reader.height, isro_reader.width
        print(f"ISRO Native Shape: {isro_w}x{isro_h}")
        
        # Divide into squares (e.g., 1000x1000)
        tile_size = 1000
        num_cols = isro_w // tile_size + (1 if isro_w % tile_size != 0 else 0)
        num_rows = isro_h // tile_size + (1 if isro_h % tile_size != 0 else 0)
        
        print(f"Dividing into {num_rows}x{num_cols} = {num_rows * num_cols} square tiles of size {tile_size}x{tile_size}...")
        
        for r in range(num_rows):
            y = r * tile_size
            h = min(tile_size, isro_h - y)
            
            # Read an entire horizontal strip of height 'h'
            row_chunk = np.array(isro_reader.mm[y:y+h, :])
            row_chunk_8bit = normalize_16bit_to_8bit(row_chunk)
            
            # Divide the horizontal strip into squares
            for c in range(num_cols):
                x = c * tile_size
                w = min(tile_size, isro_w - x)
                
                tile = row_chunk_8bit[:, x:x+w]
                tile_path = os.path.join(output_dir, f"isro_square_{r:03d}_{c:03d}.png")
                cv2.imwrite(tile_path, tile, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                
            if r % 10 == 0 or r == num_rows - 1:
                print(f"Processed row {r+1}/{num_rows} (saved {num_cols} squares)")
            
    print("Finished splitting ISRO image into squares!")

if __name__ == "__main__":
    split_isro_image()
