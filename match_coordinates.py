import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_isro_xml(xml_path):
    """Parse ISRO PDS4 XML file to extract bounding box."""
    # Namespace used in Chandrayaan-2 PDS4 files
    ns = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # We try to get Refined Corner Coordinates first, then fallback to System Level
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
            return {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons),
                'file': str(xml_path)
            }
    except Exception as e:
        # Some XML files might not be image labels (e.g. browse metadata)
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
                
                # Stop reading once we found all 4 bounds to save time
                if len(bounds) == 4:
                    break
                    
        if len(bounds) == 4:
            bounds['file'] = str(lbl_path)
            return bounds
    except Exception as e:
        pass
    return None

def check_overlap(box1, box2):
    """Check if two bounding boxes overlap or intersect."""
    lat_overlap = (box1['max_lat'] >= box2['min_lat']) and (box1['min_lat'] <= box2['max_lat'])
    lon_overlap = (box1['max_lon'] >= box2['min_lon']) and (box1['min_lon'] <= box2['max_lon'])
    return lat_overlap and lon_overlap

def main():
    isro_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO"
    jaxa_dir = "/home/friday/helios1/data"
    
    print(f"Searching for ISRO metadata in {isro_dir}...")
    isro_files = list(Path(isro_dir).rglob('*.xml'))
    print(f"Found {len(isro_files)} potential ISRO XML files.")
    
    isro_bounds = []
    for f in isro_files:
        b = parse_isro_xml(f)
        if b:
            isro_bounds.append(b)
            
    print(f"Successfully parsed {len(isro_bounds)} ISRO bounding boxes.\n")
    
    print(f"Searching for JAXA metadata in {jaxa_dir}...")
    jaxa_files = list(Path(jaxa_dir).rglob('*.lbl'))
    print(f"Found {len(jaxa_files)} potential JAXA LBL files.")
    
    jaxa_bounds = []
    for f in jaxa_files:
        b = parse_jaxa_lbl(f)
        if b:
            jaxa_bounds.append(b)
            
    print(f"Successfully parsed {len(jaxa_bounds)} JAXA bounding boxes.\n")
    
    print("--- Finding Matching Coordinates ---")
    match_count = 0
    for isro in isro_bounds:
        for jaxa in jaxa_bounds:
            if check_overlap(isro, jaxa):
                match_count += 1
                isro_name = Path(isro['file']).name
                jaxa_name = Path(jaxa['file']).name
                
                print(f"[ MATCH {match_count} ]")
                print(f"ISRO: {isro_name}")
                print(f"      Lat: {isro['min_lat']:.2f} to {isro['max_lat']:.2f} | Lon: {isro['min_lon']:.2f} to {isro['max_lon']:.2f}")
                print(f"JAXA: {jaxa_name}")
                print(f"      Lat: {jaxa['min_lat']:.2f} to {jaxa['max_lat']:.2f} | Lon: {jaxa['min_lon']:.2f} to {jaxa['max_lon']:.2f}")
                print(f"ISRO Path: {isro['file']}")
                print(f"JAXA Path: {jaxa['file']}\n")
                
    print(f"Total overlapping image pairs found: {match_count}")

if __name__ == '__main__':
    main()
