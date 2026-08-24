import os
from pathlib import Path
import cv2

# Import custom readers and geographic utilities
from match_coordinates import parse_isro_xml, parse_jaxa_lbl, check_overlap
from src.data import ISROReader, JAXAReader, normalize_16bit_to_8bit

def main():
    # File Paths
    isro_img_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.img"
    isro_xml_path = "/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"
    jaxa_dir = "/home/friday/helios1/data/selene_overlapping_data"
    
    # Output directory for the extracted patches
    output_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO/mosaics/extracted_overlaps"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Parsing ISRO Metadata...")
    isro_bounds = parse_isro_xml(isro_xml_path)
    if not isro_bounds:
        print("Failed to parse ISRO bounds.")
        return
        
    print(f"ISRO Bounds: Lat({isro_bounds['min_lat']:.4f} to {isro_bounds['max_lat']:.4f}), Lon({isro_bounds['min_lon']:.4f} to {isro_bounds['max_lon']:.4f})")
    
    # Open the ISRO reader to access full dimensions and memory map
    with ISROReader(isro_img_path, isro_xml_path) as isro_reader:
        isro_h, isro_w = isro_reader.height, isro_reader.width
        
        # Search all JAXA labels
        jaxa_lbls = list(Path(jaxa_dir).rglob('*_img.lbl'))
        print(f"\nFound {len(jaxa_lbls)} JAXA LBL files to check.")
        
        count = 0
        for lbl_path in jaxa_lbls:
            jaxa_bounds = parse_jaxa_lbl(lbl_path)
            if not jaxa_bounds:
                continue
                
            # Skip invalid wraps (like 359 to 0) that break simple bounding boxes
            if jaxa_bounds['max_lon'] - jaxa_bounds['min_lon'] > 180:
                continue 
                
            if check_overlap(isro_bounds, jaxa_bounds):
                jaxa_img_path = str(lbl_path).replace('.lbl', '.img')
                if not os.path.exists(jaxa_img_path) or os.path.getsize(jaxa_img_path) < 1000000:
                    continue
                    
                count += 1
                name = lbl_path.stem.replace('_img', '')
                print(f"\n--- [MATCH {count}] Overlap Found: {name} ---")
                
                # Compute Geographic Overlap Area
                overlap_min_lat = max(isro_bounds['min_lat'], jaxa_bounds['min_lat'])
                overlap_max_lat = min(isro_bounds['max_lat'], jaxa_bounds['max_lat'])
                overlap_min_lon = max(isro_bounds['min_lon'], jaxa_bounds['min_lon'])
                overlap_max_lon = min(isro_bounds['max_lon'], jaxa_bounds['max_lon'])
                
                center_lat = (overlap_min_lat + overlap_max_lat) / 2
                center_lon = (overlap_min_lon + overlap_max_lon) / 2
                
                # ==========================================
                # Map to ISRO Pixel Space and Extract
                # ==========================================
                # X maps from longitude, Y maps from latitude (max_lat is Y=0)
                isro_cx = int((center_lon - isro_bounds['min_lon']) / (isro_bounds['max_lon'] - isro_bounds['min_lon']) * isro_w)
                isro_cy = int((isro_bounds['max_lat'] - center_lat) / (isro_bounds['max_lat'] - isro_bounds['min_lat']) * isro_h)
                
                # Map geographic span to pixel size
                isro_span_h = int((overlap_max_lat - overlap_min_lat) / (isro_bounds['max_lat'] - isro_bounds['min_lat']) * isro_h)
                isro_span_w = int((overlap_max_lon - overlap_min_lon) / (isro_bounds['max_lon'] - isro_bounds['min_lon']) * isro_w)
                
                # Clamp size between 1000 and 8000 to prevent OOM
                isro_size = max(isro_span_h, isro_span_w)
                isro_size = min(max(isro_size, 1000), 8000)
                
                print(f"Extracting from ISRO at CX={isro_cx}, CY={isro_cy}, Size={isro_size}")
                isro_patch_raw = isro_reader.extract_patch(isro_cx, isro_cy, isro_size)
                isro_patch_8bit = normalize_16bit_to_8bit(isro_patch_raw)
                
                isro_save_path = os.path.join(output_dir, f"overlap_{name}_ISRO.png")
                cv2.imwrite(isro_save_path, isro_patch_8bit)
                print(f"Saved: {isro_save_path}")
                
                
                with JAXAReader(jaxa_img_path, str(lbl_path)) as jaxa_reader:
                    jaxa_h, jaxa_w = jaxa_reader.height, jaxa_reader.width
                    
                    jaxa_cx = int((center_lon - jaxa_bounds['min_lon']) / (jaxa_bounds['max_lon'] - jaxa_bounds['min_lon']) * jaxa_w)
                    jaxa_cy = int((jaxa_bounds['max_lat'] - center_lat) / (jaxa_bounds['max_lat'] - jaxa_bounds['min_lat']) * jaxa_h)
                    
                    jaxa_span_h = int((overlap_max_lat - overlap_min_lat) / (jaxa_bounds['max_lat'] - jaxa_bounds['min_lat']) * jaxa_h)
                    jaxa_span_w = int((overlap_max_lon - overlap_min_lon) / (jaxa_bounds['max_lon'] - jaxa_bounds['min_lon']) * jaxa_w)
                    
                    jaxa_size = max(jaxa_span_h, jaxa_span_w)
                    jaxa_size = min(max(jaxa_size, 1000), 8000)
                    
                    print(f"Extracting from JAXA at CX={jaxa_cx}, CY={jaxa_cy}, Size={jaxa_size}")
                    jaxa_patch_raw = jaxa_reader.extract_patch(jaxa_cx, jaxa_cy, jaxa_size)
                    jaxa_patch_8bit = normalize_16bit_to_8bit(jaxa_patch_raw)
                    
                    jaxa_save_path = os.path.join(output_dir, f"overlap_{name}_JAXA.png")
                    cv2.imwrite(jaxa_save_path, jaxa_patch_8bit)
                    print(f"Saved: {jaxa_save_path}")
                
        print(f"\nDone! Extracted {count} pairs of overlapping images.")
        print(f"Images are saved in: {output_dir}")

if __name__ == '__main__':
    main()
