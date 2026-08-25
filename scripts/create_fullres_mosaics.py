import os
import cv2
import numpy as np
from pathlib import Path
from match_coordinates import parse_isro_xml, parse_jaxa_lbl
from src.data import ISROReader, JAXAReader, normalize_16bit_to_8bit

def create_fullres_mosaics():
    isro_img_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.img"
    isro_xml_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"
    
    jaxa_dir = "/home/friday/helios1/data/selene_overlapping_data"
    output_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO/mosaics"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Processing ISRO OHRC Image at FULL NATIVE RESOLUTION...")
    with ISROReader(isro_img_path, isro_xml_path) as isro_reader:
        isro_h, isro_w = isro_reader.height, isro_reader.width
        print(f"ISRO Native Shape: {isro_w}x{isro_h}")
        
        # Allocate full resolution canvas
        isro_canvas = np.zeros((isro_h, isro_w), dtype=np.uint8)
        
        chunk_size = 5000
        for y in range(0, isro_h, chunk_size):
            h = min(chunk_size, isro_h - y)
            chunk = np.array(isro_reader.mm[y:y+h, :])
            chunk_8bit = normalize_16bit_to_8bit(chunk)
            
            isro_canvas[y:y+h, :] = chunk_8bit
            if y % 20000 == 0:
                print(f"   Processed ISRO chunk {y}/{isro_h}")
            
    isro_out = os.path.join(output_dir, "isro_fullres.png")
    print(f"Saving {isro_out} (this might take a while due to massive file size)...")
    cv2.imwrite(isro_out, isro_canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1]) # Low compression for speed
    print("Saved isro_fullres.png!")
    
    # Free memory before JAXA
    del isro_canvas
    
    # Parse ISRO bounds first to use as canvas limits
    isro_bounds = parse_isro_xml(isro_xml_path)
    print(f"ISRO Bounds: {isro_bounds}")
    
    # We will pad the ISRO bounds slightly (e.g. 0.5 degrees) for the JAXA canvas
    # to provide a bit of surrounding context.
    pad = 0.5
    canvas_min_lat = isro_bounds['min_lat'] - pad
    canvas_max_lat = isro_bounds['max_lat'] + pad
    canvas_min_lon = isro_bounds['min_lon'] - pad
    canvas_max_lon = isro_bounds['max_lon'] + pad

    print("\n2. Processing JAXA Images into FULL RES Mosaic...")
    jaxa_lbls = list(Path(jaxa_dir).rglob('*_img.lbl'))
    jaxa_images = []
    
    for lbl in jaxa_lbls:
        bounds = parse_jaxa_lbl(lbl)
        if bounds:
            # Fix longitude wrap-around (if max_lon - min_lon > 180, it crossed prime meridian)
            if bounds['max_lon'] - bounds['min_lon'] > 180:
                continue
                
            # Check overlap with our canvas bounds
            is_overlapping = not (bounds['max_lon'] < canvas_min_lon or 
                                  bounds['min_lon'] > canvas_max_lon or 
                                  bounds['max_lat'] < canvas_min_lat or 
                                  bounds['min_lat'] > canvas_max_lat)
            
            if is_overlapping:
                img_path = str(lbl).replace('.lbl', '.img')
                if os.path.exists(img_path) and os.path.getsize(img_path) > 1000000:
                    jaxa_images.append({'lbl': lbl, 'img': img_path, 'bounds': bounds})
                
    if not jaxa_images:
        print("No overlapping JAXA images found.")
        return
        
    print(f"Found {len(jaxa_images)} JAXA images to mosaic.")
    
    # Calculate native resolution PPD
    # JAXA SELENE TC Ortho is roughly ~10m/px, which translates to ~3030 pixels per degree on the moon.
    PPD = 3030
    canvas_h = int((canvas_max_lat - canvas_min_lat) * PPD)
    canvas_w = int((canvas_max_lon - canvas_min_lon) * PPD)
    
    print(f"Creating FULL RES JAXA canvas of size {canvas_w}x{canvas_h}")
    # Allocate empty canvas
    jaxa_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    for i, jdata in enumerate(jaxa_images):
        print(f"   Mosaicing JAXA {i+1}/{len(jaxa_images)}: {Path(jdata['img']).name}")
        try:
            with JAXAReader(jdata['img'], str(jdata['lbl'])) as reader:
                img_data = reader.read_all()
                img_8bit = normalize_16bit_to_8bit(img_data)
                
                b = jdata['bounds']
                y_offset = int((canvas_max_lat - b['max_lat']) * PPD)
                x_offset = int((b['min_lon'] - canvas_min_lon) * PPD)
                
                w = int((b['max_lon'] - b['min_lon']) * PPD)
                h = int((b['max_lat'] - b['min_lat']) * PPD)
                
                if w <= 0 or h <= 0: continue
                
                # Resize to map precisely into the PPD grid
                resized = cv2.resize(img_8bit, (w, h), interpolation=cv2.INTER_LANCZOS4)
                
                # Determine overlap with canvas
                y1 = max(0, y_offset)
                y2 = min(canvas_h, y_offset + h)
                x1 = max(0, x_offset)
                x2 = min(canvas_w, x_offset + w)
                
                if y1 >= y2 or x1 >= x2:
                    continue
                    
                # Crop the resized image to match the canvas region
                crop_y1 = y1 - y_offset
                crop_y2 = y2 - y_offset
                crop_x1 = x1 - x_offset
                crop_x2 = x2 - x_offset
                
                cropped = resized[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Blend
                jaxa_canvas[y1:y2, x1:x2] = np.maximum(
                    jaxa_canvas[y1:y2, x1:x2], 
                    cropped
                )
        except Exception as e:
            print(f"Failed to process {jdata['img']}: {e}")
            
    jaxa_out = os.path.join(output_dir, "jaxa_fullres.png")
    print(f"Saving {jaxa_out} (this might take a while)...")
    cv2.imwrite(jaxa_out, jaxa_canvas, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    print("Saved jaxa_fullres.png!")

if __name__ == '__main__':
    create_fullres_mosaics()
