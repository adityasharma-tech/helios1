import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_isro_xml(xml_path):
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
            
            # Normalize longitude to 0-360 just in case there are negative longitudes
            lons = [lon if lon >= 0 else lon + 360 for lon in lons]
            
            return {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons)
            }
    except Exception:
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
                    break
                    
        if len(bounds) == 4:
            # Normalize longitude to 0-360
            bounds['min_lon'] = bounds['min_lon'] if bounds['min_lon'] >= 0 else bounds['min_lon'] + 360
            bounds['max_lon'] = bounds['max_lon'] if bounds['max_lon'] >= 0 else bounds['max_lon'] + 360
            return bounds
    except Exception:
        pass
    return None

def main():
    isro_dir = "/media/friday/Toshiba Drive/FILES/helios/ISRO"
    jaxa_dir = "/home/friday/helios1/data"
    
    print("Parsing ISRO files...")
    isro_bounds = [b for b in [parse_isro_xml(f) for f in Path(isro_dir).rglob('*.xml')] if b]
    
    print("Parsing JAXA files...")
    jaxa_bounds = [b for b in [parse_jaxa_lbl(f) for f in Path(jaxa_dir).rglob('*.lbl')] if b]
    
    print("Plotting map...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot JAXA Bounds in Blue
    for idx, b in enumerate(jaxa_bounds):
        rect = patches.Rectangle(
            (b['min_lon'], b['min_lat']), 
            b['max_lon'] - b['min_lon'], 
            b['max_lat'] - b['min_lat'],
            linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3,
            label='JAXA (SELENE)' if idx == 0 else ""
        )
        ax.add_patch(rect)
        
    # Plot ISRO Bounds in Red
    for idx, b in enumerate(isro_bounds):
        rect = patches.Rectangle(
            (b['min_lon'], b['min_lat']), 
            b['max_lon'] - b['min_lon'], 
            b['max_lat'] - b['min_lat'],
            linewidth=2, edgecolor='red', facecolor='red', alpha=0.8,
            label='ISRO (Chandrayaan-2)' if idx == 0 else ""
        )
        ax.add_patch(rect)

    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title('Spatial Coverage: ISRO vs JAXA Data on Lunar Map')
    
    # Add simple grid lines to represent equator and prime meridian
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(180, color='black', linewidth=0.5, linestyle='--')
    
    ax.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "/home/friday/helios1/map_coverage.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Map successfully saved to: {output_path}")

if __name__ == '__main__':
    main()
