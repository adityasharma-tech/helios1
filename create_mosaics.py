import os
import cv2
import numpy as np
from pathlib import Path
from match_coordinates import parse_isro_xml, parse_jaxa_lbl
from src.data import ISROReader, JAXAReader, normalize_16bit_to_8bit

def create_mosaics():
    isro_img_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.img"
    isro_xml_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"
    
    jaxa_dir = "/home/friday/helios1/data/selene_overlapping_data"
    
    print("1. Downsampling ISRO OHRC Image...")
    with ISROReader(isro_img_path, isro_xml_path) as isro_reader:
        isro_h, isro_w = isro_reader.height, isro_reader.width
        print(f"ISRO Shape: {isro_w}x{isro_h}")
        
        # Determine scale factor to fit height in ~3000px so it fits in memory easily
        scale_factor = 3000.0 / max(1, isro_h)
        if scale_factor > 1: scale_factor = 1.0
        
        isro_small_h = int(isro_h * scale_factor)
        isro_small_w = int(isro_w * scale_factor)
        
        isro_canvas = np.zeros((isro_small_h, isro_small_w), dtype=np.uint8)
        
        chunk_size = 10000
        for y in range(0, isro_h, chunk_size):
            h = min(chunk_size, isro_h - y)
            chunk = np.array(isro_reader.mm[y:y+h, :])
            chunk_8bit = normalize_16bit_to_8bit(chunk)
            
            new_h = int(h * scale_factor)
            if new_h == 0: continue
            
            chunk_resized = cv2.resize(chunk_8bit, (isro_small_w, new_h), interpolation=cv2.INTER_AREA)
            
            y_dst = int(y * scale_factor)
            if y_dst + new_h > isro_small_h:
                new_h = isro_small_h - y_dst
                chunk_resized = chunk_resized[:new_h, :]
                
            isro_canvas[y_dst:y_dst+new_h, :] = chunk_resized
            print(f"   Processed ISRO chunk {y}/{isro_h}")
            
    cv2.imwrite("isro_mosaic.png", isro_canvas)
    print("Saved isro_mosaic.png")

    print("\n2. Processing JAXA Images into Mosaic...")
    jaxa_lbls = list(Path(jaxa_dir).rglob('*_img.lbl'))
    jaxa_images = []
    
    min_lat, max_lat = float('inf'), float('-inf')
    min_lon, max_lon = float('inf'), float('-inf')
    
    for lbl in jaxa_lbls:
        bounds = parse_jaxa_lbl(lbl)
        if bounds:
            img_path = str(lbl).replace('.lbl', '.img')
            # Only process images that have actually downloaded successfully (e.g. >1MB)
            if os.path.exists(img_path) and os.path.getsize(img_path) > 1000000:
                jaxa_images.append({'lbl': lbl, 'img': img_path, 'bounds': bounds})
                min_lat = min(min_lat, bounds['min_lat'])
                max_lat = max(max_lat, bounds['max_lat'])
                min_lon = min(min_lon, bounds['min_lon'])
                max_lon = max(max_lon, bounds['max_lon'])
                
    if not jaxa_images:
        print("No JAXA images downloaded yet.")
        return
        
    print(f"Found {len(jaxa_images)} JAXA images to mosaic.")
    
    PPD = 2000
    canvas_h = int((max_lat - min_lat) * PPD)
    canvas_w = int((max_lon - min_lon) * PPD)
    
    # Cap dimensions to avoid memory issues
    if canvas_h > 8000 or canvas_w > 8000:
        scale = 8000 / max(canvas_h, canvas_w)
        PPD = PPD * scale
        canvas_h = int((max_lat - min_lat) * PPD)
        canvas_w = int((max_lon - min_lon) * PPD)
        
    print(f"Creating JAXA canvas of size {canvas_w}x{canvas_h}")
    jaxa_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    for i, jdata in enumerate(jaxa_images):
        print(f"   Mosaicing JAXA {i+1}/{len(jaxa_images)}: {Path(jdata['img']).name}")
        try:
            with JAXAReader(jdata['img'], str(jdata['lbl'])) as reader:
                img_data = reader.read_all()
                img_8bit = normalize_16bit_to_8bit(img_data)
                
                b = jdata['bounds']
                y1 = int((max_lat - b['max_lat']) * PPD)
                y2 = int((max_lat - b['min_lat']) * PPD)
                x1 = int((b['min_lon'] - min_lon) * PPD)
                x2 = int((b['max_lon'] - min_lon) * PPD)
                
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                
                resized = cv2.resize(img_8bit, (w, h), interpolation=cv2.INTER_AREA)
                
                h_actual = min(h, canvas_h - y1)
                w_actual = min(w, canvas_w - x1)
                resized = resized[:h_actual, :w_actual]
                
                jaxa_canvas[y1:y1+h_actual, x1:x1+w_actual] = np.maximum(
                    jaxa_canvas[y1:y1+h_actual, x1:x1+w_actual], 
                    resized
                )
        except Exception as e:
            print(f"Failed to process {jdata['img']}: {e}")
            
    cv2.imwrite("jaxa_mosaic.png", jaxa_canvas)
    print("Saved jaxa_mosaic.png")

if __name__ == '__main__':
    create_mosaics()
