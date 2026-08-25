import os
import re
import urllib.request
from urllib.parse import urljoin
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://data.darts.isas.jaxa.jp/pub/pds3/sln-l-tc-4-dtm-ortho-v3.0/"
OUTPUT_DIR = "./selene_metadata"

def parse_isro_xml(xml_path):
    """Parse ISRO PDS4 XML file to extract bounding box."""
    ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        coords = root.find('.//isda:Refined_Corner_Coordinates', ns)
        if coords is None:
            coords = root.find('.//isda:System_Level_Coordinates', ns)
        if coords is not None:
            lats = [
                float(coords.find('isda:upper_left_latitude', ns).text),
                float(coords.find('isda:upper_right_latitude', ns).text),
                float(coords.find('isda:lower_left_latitude', ns).text),
                float(coords.find('isda:lower_right_latitude', ns).text)
            ]
            lons = [
                float(coords.find('isda:upper_left_longitude', ns).text),
                float(coords.find('isda:upper_right_longitude', ns).text),
                float(coords.find('isda:lower_left_longitude', ns).text),
                float(coords.find('isda:lower_right_longitude', ns).text)
            ]
            # Normalize longitude to 0-360
            lons = [lon if lon >= 0 else lon + 360 for lon in lons]
            return {'min_lat': min(lats), 'max_lat': max(lats), 'min_lon': min(lons), 'max_lon': max(lons)}
    except:
        pass
    return None

def parse_jaxa_lbl(lbl_path):
    """Parse JAXA PDS3 LBL file to extract bounding box."""
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
    """Check if two bounding boxes overlap, given a certain degree buffer radius."""
    lat_overlap = (box1['max_lat'] + buffer_deg >= box2['min_lat']) and (box1['min_lat'] - buffer_deg <= box2['max_lat'])
    lon_overlap = (box1['max_lon'] + buffer_deg >= box2['min_lon']) and (box1['min_lon'] - buffer_deg <= box2['max_lon'])
    return lat_overlap and lon_overlap

def get_links(url):
    """Scrape the Apache directory listing to get href links."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            links = re.findall(r'href="([^"]+)"', html)
            return [l for l in links if not l.startswith('?') and not l.startswith('/') and l != '../']
    except:
        return []

def download_file(url, dest_path):
    """Download a file if it doesn't already exist."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(dest_path, 'wb') as f:
                f.write(response.read())
    except:
        pass

def download_entire_directory(folder):
    """Scrapes the remote folder and downloads ALL .lbl files in parallel."""
    print(f"\n--- Downloading ENTIRE data directory for: {folder} ---")
    folder_url = urljoin(BASE_URL, folder)
    data_url = urljoin(folder_url, "data/")
    data_links = get_links(data_url)
    
    lbl_files = [l for l in data_links if l.endswith('.lbl')]
    print(f"Found {len(lbl_files)} .lbl files in {folder}data/. Downloading ALL of them concurrently...")
    
    def dl_task(lbl):
        file_url = urljoin(data_url, lbl)
        dest_path = os.path.join(OUTPUT_DIR, folder, "data", lbl)
        download_file(file_url, dest_path)
        
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(dl_task, lbl) for lbl in lbl_files]
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 250 == 0:
                print(f"[{folder}] Progress: {completed} / {len(lbl_files)} labels downloaded...")
    
    print(f"--- Finished downloading {folder} ---")

def main():
    isro_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO"
    jaxa_dir = "/home/friday/helios1/data"
    
    print("1. Parsing ISRO Bounds...")
    isro_files = list(Path(isro_dir).rglob('*.xml'))
    isro_bounds = []
    for f in isro_files:
        b = parse_isro_xml(f)
        if b: isro_bounds.append(b)
        
    if not isro_bounds:
        print("No ISRO bounds found!")
        return
        
    # Combine all ISRO bounding boxes into one massive bounding box for the general ISRO area
    global_isro_box = {
        'min_lat': min(b['min_lat'] for b in isro_bounds),
        'max_lat': max(b['max_lat'] for b in isro_bounds),
        'min_lon': min(b['min_lon'] for b in isro_bounds),
        'max_lon': max(b['max_lon'] for b in isro_bounds)
    }
    
    print(f"ISRO General Area: Lat({global_isro_box['min_lat']:.2f} to {global_isro_box['max_lat']:.2f}), Lon({global_isro_box['min_lon']:.2f} to {global_isro_box['max_lon']:.2f})")
    
    print("\n2. Scanning local JAXA files to find nearby directories...")
    jaxa_files = list(Path(jaxa_dir).rglob('*.lbl'))
    
    matching_folders = set()
    # A 12-degree buffer covers a wide radius to catch the "nearby" blue blobs
    buffer_deg = 12 
    
    for f in jaxa_files:
        b = parse_jaxa_lbl(f)
        if b and check_overlap_with_buffer(global_isro_box, b, buffer_deg):
            # Extract the folder name (e.g., "20071225/") from the local path
            parts = Path(f).parts
            if "selene_metadata" in parts:
                idx = parts.index("selene_metadata")
                if idx + 1 < len(parts):
                    folder = parts[idx + 1] + "/"
                    matching_folders.add(folder)
                    
    print(f"\nFound {len(matching_folders)} JAXA directories that contain data near ISRO bounds:")
    for m in matching_folders:
        print(f" - {m}")
        
    if not matching_folders:
        print("No nearby JAXA data found locally. Wait for your previous downloads to pull more data, or increase the buffer.")
        return
        
    print("\n3. Downloading FULL labels for matching directories...")
    # Process each matched folder one by one, downloading ALL of its labels
    for folder in matching_folders:
        download_entire_directory(folder)
        
    print("\nAll targeted nearby directories have been fully downloaded!")

if __name__ == '__main__':
    main()
