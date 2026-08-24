import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Import existing readers and utilities
from match_coordinates import parse_isro_xml, parse_jaxa_lbl, check_overlap
from src.data import ISROReader, JAXAReader, normalize_16bit_to_8bit

def get_overlap(box1, box2):
    """Calculate the geographic overlap between two bounding boxes."""
    min_lat = max(box1['min_lat'], box2['min_lat'])
    max_lat = min(box1['max_lat'], box2['max_lat'])
    min_lon = max(box1['min_lon'], box2['min_lon'])
    max_lon = min(box1['max_lon'], box2['max_lon'])
    
    if min_lat >= max_lat or min_lon >= max_lon:
        return None
    return {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon}

def process_dataset(dataset_name, files, target_box, canvas, reader_class):
    target_min_lat = target_box['min_lat']
    target_max_lat = target_box['max_lat']
    target_min_lon = target_box['min_lon']
    target_max_lon = target_box['max_lon']
    canvas_h, canvas_w = canvas.shape
    
    for f in files:
        if dataset_name == 'ISRO':
            bounds = parse_isro_xml(f)
            img_path = str(f).replace('.xml', '.img')
        else:
            bounds = parse_jaxa_lbl(f)
            img_path = str(f).replace('.lbl', '.img')
            
        if not bounds or not os.path.exists(img_path):
            continue
            
        # Avoid division by zero
        if bounds['max_lon'] <= bounds['min_lon'] or bounds['max_lat'] <= bounds['min_lat']:
            continue
            
        # Handle wraparound
        if bounds['max_lon'] - bounds['min_lon'] > 180:
            continue
            
        overlap = get_overlap(target_box, bounds)
        if not overlap:
            continue
            
        print(f"[{dataset_name}] Found overlapping image: {f.name}")
        
        try:
            with reader_class(img_path, str(f)) as reader:
                img_h, img_w = reader.height, reader.width
                
                # Image pixel coordinates for the overlap region
                # X maps from min_lon (0) to max_lon (img_w)
                img_x1 = int((overlap['min_lon'] - bounds['min_lon']) / (bounds['max_lon'] - bounds['min_lon']) * img_w)
                img_x2 = int((overlap['max_lon'] - bounds['min_lon']) / (bounds['max_lon'] - bounds['min_lon']) * img_w)
                
                # Y maps from max_lat (0) to min_lat (img_h) because images draw from top to bottom
                img_y1 = int((bounds['max_lat'] - overlap['max_lat']) / (bounds['max_lat'] - bounds['min_lat']) * img_h)
                img_y2 = int((bounds['max_lat'] - overlap['min_lat']) / (bounds['max_lat'] - bounds['min_lat']) * img_h)
                
                # Clamp to image boundaries
                img_x1, img_x2 = max(0, min(img_w, img_x1)), max(0, min(img_w, img_x2))
                img_y1, img_y2 = max(0, min(img_h, img_y1)), max(0, min(img_h, img_y2))
                
                if img_x2 <= img_x1 or img_y2 <= img_y1:
                    continue
                    
                # Canvas pixel coordinates for the overlap region
                can_x1 = int((overlap['min_lon'] - target_min_lon) / (target_max_lon - target_min_lon) * canvas_w)
                can_x2 = int((overlap['max_lon'] - target_min_lon) / (target_max_lon - target_min_lon) * canvas_w)
                can_y1 = int((target_max_lat - overlap['max_lat']) / (target_max_lat - target_min_lat) * canvas_h)
                can_y2 = int((target_max_lat - overlap['min_lat']) / (target_max_lat - target_min_lat) * canvas_h)
                
                can_x1, can_x2 = max(0, min(canvas_w, can_x1)), max(0, min(canvas_w, can_x2))
                can_y1, can_y2 = max(0, min(canvas_h, can_y1)), max(0, min(canvas_h, can_y2))
                
                if can_x2 <= can_x1 or can_y2 <= can_y1:
                    continue
                
                # Calculate step size to downsample directly from the memmap (saves RAM and time)
                step_y = max(1, (img_y2 - img_y1) // (can_y2 - can_y1))
                step_x = max(1, (img_x2 - img_x1) // (can_x2 - can_x1))
                
                # Read the subsampled patch
                patch_raw = np.array(reader.mm[img_y1:img_y2:step_y, img_x1:img_x2:step_x])
                patch_8bit = normalize_16bit_to_8bit(patch_raw)
                
                # Resize to perfectly fit the canvas coordinates
                patch_resized = cv2.resize(patch_8bit, (can_x2 - can_x1, can_y2 - can_y1), interpolation=cv2.INTER_AREA)
                
                # Blend with existing canvas (take maximum pixel value to combine multiple overlapping strips)
                canvas[can_y1:can_y2, can_x1:can_x2] = np.maximum(canvas[can_y1:can_y2, can_x1:can_x2], patch_resized)
                
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract a specific lat/lon regional map from ISRO and JAXA")
    parser.add_argument('--min-lon', type=float, default=135.0)
    parser.add_argument('--max-lon', type=float, default=150.0)
    parser.add_argument('--min-lat', type=float, default=-40.0)
    parser.add_argument('--max-lat', type=float, default=0.0)
    parser.add_argument('--height', type=int, default=1000, help="Height of the output image in pixels")
    args = parser.parse_args()
    
    target_box = {
        'min_lon': args.min_lon,
        'max_lon': args.max_lon,
        'min_lat': args.min_lat,
        'max_lat': args.max_lat
    }
    
    if args.max_lat <= args.min_lat or args.max_lon <= args.min_lon:
        print("Invalid coordinates! Max must be greater than Min.")
        return

    lon_span = args.max_lon - args.min_lon
    lat_span = args.max_lat - args.min_lat
    
    # Calculate width based on geographic aspect ratio
    canvas_h = args.height
    canvas_w = int(canvas_h * (lon_span / lat_span))
    
    print(f"Target Region: Lon {args.min_lon} to {args.max_lon}, Lat {args.min_lat} to {args.max_lat}")
    print(f"Output Canvas Size per dataset: {canvas_w}x{canvas_h}")
    
    # Create blank canvases
    isro_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    jaxa_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    
    isro_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO"
    jaxa_dir = "/home/friday/helios1/data/selene_overlapping_data"
    
    isro_files = list(Path(isro_dir).rglob('*.xml'))
    jaxa_files = list(Path(jaxa_dir).rglob('*.lbl'))
    
    print(f"\nSearching {len(isro_files)} ISRO metadata files...")
    process_dataset('ISRO', isro_files, target_box, isro_canvas, ISROReader)
    
    print(f"\nSearching {len(jaxa_files)} JAXA metadata files...")
    process_dataset('JAXA', jaxa_files, target_box, jaxa_canvas, JAXAReader)
    
    # Convert to color to add labels and separator
    isro_bgr = cv2.cvtColor(isro_canvas, cv2.COLOR_GRAY2BGR)
    jaxa_bgr = cv2.cvtColor(jaxa_canvas, cv2.COLOR_GRAY2BGR)
    
    # Add text labels
    cv2.putText(isro_bgr, f"ISRO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(jaxa_bgr, f"JAXA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    label = f"Lon: {args.min_lon} to {args.max_lon} | Lat: {args.min_lat} to {args.max_lat}"
    cv2.putText(isro_bgr, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(jaxa_bgr, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    
    # Create a 5-pixel wide white separator line
    separator = np.ones((canvas_h, 5, 3), dtype=np.uint8) * 255
    
    # Combine horizontally
    final_img = np.hstack((isro_bgr, separator, jaxa_bgr))
    
    output_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO/mosaics"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"regional_map_{args.min_lon}_{args.max_lon}_{args.min_lat}_{args.max_lat}".replace('.', '_') + ".png"
    output_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(output_path, final_img)
    print(f"\nSuccess! Saved regional map side-by-side to:")
    print(output_path)

if __name__ == '__main__':
    main()
