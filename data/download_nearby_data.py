import os
import re
import urllib.request
from urllib.parse import urljoin
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-tc-4-dtm-ortho-v3.0/"
LOCAL_METADATA_DIR = "/home/friday/helios1/data/selene_metadata"
OUTPUT_DIR = "/home/friday/helios1/data/selene_overlapping_data"

def parse_isro_xml(xml_path):
    ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        coords = root.find('.//isda:Refined_Corner_Coordinates', ns)
        if coords is None:
            coords = root.find('.//isda:System_Level_Coordinates', ns)
        if coords is not None:
            lats = [float(coords.find('isda:upper_left_latitude', ns).text), float(coords.find('isda:lower_right_latitude', ns).text)]
            lons = [float(coords.find('isda:upper_left_longitude', ns).text), float(coords.find('isda:lower_right_longitude', ns).text)]
            lons = [lon if lon >= 0 else lon + 360 for lon in lons]
            return {'min_lat': min(lats), 'max_lat': max(lats), 'min_lon': min(lons), 'max_lon': max(lons)}
    except:
        pass
    return None

def parse_jaxa_lbl(lbl_path):
    bounds = {}
    try:
        with open(lbl_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'MAXIMUM_LATITUDE' in line:
                    match = re.search(r'([-+]?\d*\.\d+|\d+)', line)
                    if match: bounds['max_lat'] = float(match.group())
                elif 'MINIMUM_LATITUDE' in line:
                    match = re.search(r'([-+]?\d*\.\d+|\d+)', line)
                    if match: bounds['min_lat'] = float(match.group())
                elif 'EASTERNMOST_LONGITUDE' in line:
                    match = re.search(r'([-+]?\d*\.\d+|\d+)', line)
                    if match: bounds['max_lon'] = float(match.group())
                elif 'WESTERNMOST_LONGITUDE' in line:
                    match = re.search(r'([-+]?\d*\.\d+|\d+)', line)
                    if match: bounds['min_lon'] = float(match.group())
                if len(bounds) == 4:
                    bounds['min_lon'] = bounds['min_lon'] if bounds['min_lon'] >= 0 else bounds['min_lon'] + 360
                    bounds['max_lon'] = bounds['max_lon'] if bounds['max_lon'] >= 0 else bounds['max_lon'] + 360
                    return bounds
    except:
        pass
    return None

def check_overlap_with_buffer(box1, box2, buffer_deg):
    lat_overlap = (box1['max_lat'] + buffer_deg >= box2['min_lat']) and (box1['min_lat'] - buffer_deg <= box2['max_lat'])
    lon_overlap = (box1['max_lon'] + buffer_deg >= box2['min_lon']) and (box1['min_lon'] - buffer_deg <= box2['max_lon'])
    return lat_overlap and lon_overlap

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        # Only skip if it's already downloaded with a file size greater than 0
        if os.path.getsize(dest_path) > 0:
            return
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            # We use chunking so we don't load a 50MB image directly into RAM
            with open(dest_path, 'wb') as f:
                while chunk := response.read(8192):
                    f.write(chunk)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    isro_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO"
    
    print("1. Parsing ISRO Bounds...")
    isro_files = list(Path(isro_dir).rglob('*.xml'))
    isro_bounds = []
    for f in isro_files:
        b = parse_isro_xml(f)
        if b: isro_bounds.append(b)
        
    if not isro_bounds:
        print("No ISRO bounds found!")
        return
        
    global_isro_box = {
        'min_lat': min(b['min_lat'] for b in isro_bounds),
        'max_lat': max(b['max_lat'] for b in isro_bounds),
        'min_lon': min(b['min_lon'] for b in isro_bounds),
        'max_lon': max(b['max_lon'] for b in isro_bounds)
    }
    
    print(f"ISRO General Area: Lat({global_isro_box['min_lat']:.2f} to {global_isro_box['max_lat']:.2f}), Lon({global_isro_box['min_lon']:.2f} to {global_isro_box['max_lon']:.2f})")
    
    print("\n2. Finding strictly overlapping JAXA datasets...")
    jaxa_files = list(Path(LOCAL_METADATA_DIR).rglob('*.lbl'))
    
    # Very tight buffer: 2 degrees to ensure we only get overlapping or very nearby actual data
    buffer_deg = 2 
    
    # Set to store unique (folder_name, base_filename)
    targets = set()
    
    for f in jaxa_files:
        b = parse_jaxa_lbl(f)
        if b and check_overlap_with_buffer(global_isro_box, b, buffer_deg):
            parts = Path(f).parts
            if "selene_metadata" in parts:
                idx = parts.index("selene_metadata")
                if idx + 1 < len(parts):
                    folder = parts[idx + 1]
                    name = Path(f).name
                    # Extract the base dataset name, e.g., DTMTCO_03_00775N645E0849PS
                    match = re.match(r'(.*)_[a-z]+\.lbl', name)
                    if match:
                        base_name = match.group(1)
                        targets.add((folder, base_name))
                        
    print(f"\nFound {len(targets)} precise datasets highly overlapping with ISRO data.")
    for t in targets:
        print(f" - Folder: {t[0]}, Dataset: {t[1]}")
        
    if not targets:
        print("No heavily overlapping data found. Try increasing buffer or downloading more metadata.")
        return
        
    print("\n3. Downloading ACTUAL heavy data (.img) for these targets...")
    
    download_tasks = []
    for folder, base_name in targets:
        data_url_base = urljoin(BASE_URL, f"{folder}/data/")
        
        # Target the primary image files (ortho-images and digital terrain models)
        files_to_grab = [
            f"{base_name}_img.img",
            f"{base_name}_img.lbl",
            f"{base_name}_dtm.img",
            f"{base_name}_dtm.lbl"
        ]
        
        for file_name in files_to_grab:
            file_url = urljoin(data_url_base, file_name)
            dest_path = os.path.join(OUTPUT_DIR, folder, file_name)
            download_tasks.append((file_url, dest_path))
            
    print(f"Queueing {len(download_tasks)} heavy files for download...")
    
    def dl_task(task):
        url, dest = task
        print(f"Downloading: {Path(dest).name}")
        download_file(url, dest)
        return dest
        
    # We use fewer workers (5) because these .img files are large (25MB - 50MB each)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(dl_task, task) for task in download_tasks]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            print(f"Progress: {completed} / {len(download_tasks)} completed.")
            
    print("\nAll massive data files have been successfully downloaded to 'selene_overlapping_data'!")

if __name__ == '__main__':
    main()
