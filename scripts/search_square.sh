#!/bin/bash

# Activate virtualenv
source .venv/bin/activate

ISRO_IMG="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.img"
ISRO_XML="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"

JAXA_DIR="/home/friday/helios1/data/selene_overlapping_data"
OUTPUT_DIR="/media/friday/Toshiba Drive/FILES/helios/ISRO/mosaics/jaxa_matches"
mkdir -p "$OUTPUT_DIR"

# Use Python to find which JAXA images overlap with ISRO and compute
# the pixel coordinates where they overlap in the ISRO strip.
# This outputs lines like: JAXA_IMG|JAXA_LBL|CX|CY|SIZE
echo "============================================="
echo " Finding overlapping JAXA images..."
echo "============================================="

OVERLAP_LIST=$(python -c "
from match_coordinates import parse_isro_xml, parse_jaxa_lbl, check_overlap
from pathlib import Path
import os

isro_xml = '$ISRO_XML'
jaxa_dir = '$JAXA_DIR'

isro = parse_isro_xml(isro_xml)
# ISRO image dimensions
isro_h, isro_w = 101063, 12000

for lbl in sorted(Path(jaxa_dir).rglob('*_img.lbl')):
    jaxa = parse_jaxa_lbl(lbl)
    if not jaxa:
        continue
    # Skip wrap-around images
    if jaxa['max_lon'] - jaxa['min_lon'] > 180:
        continue
    if not check_overlap(isro, jaxa):
        continue
    
    img_path = str(lbl).replace('.lbl', '.img')
    if not os.path.exists(img_path) or os.path.getsize(img_path) < 1000000:
        continue
    
    # Convert overlap center from lat/lon to pixel coords in ISRO strip
    # ISRO pixel mapping: latitude runs top to bottom, longitude runs left to right
    overlap_min_lat = max(isro['min_lat'], jaxa['min_lat'])
    overlap_max_lat = min(isro['max_lat'], jaxa['max_lat'])
    overlap_min_lon = max(isro['min_lon'], jaxa['min_lon'])
    overlap_max_lon = min(isro['max_lon'], jaxa['max_lon'])
    
    overlap_center_lat = (overlap_min_lat + overlap_max_lat) / 2
    overlap_center_lon = (overlap_min_lon + overlap_max_lon) / 2
    
    # Map lat/lon to pixel coordinates
    # X (column) maps from longitude
    cx = int((overlap_center_lon - isro['min_lon']) / (isro['max_lon'] - isro['min_lon']) * isro_w)
    # Y (row) maps from latitude (top = max_lat, bottom = min_lat)
    cy = int((isro['max_lat'] - overlap_center_lat) / (isro['max_lat'] - isro['min_lat']) * isro_h)
    
    # Clamp
    cx = max(0, min(isro_w, cx))
    cy = max(0, min(isro_h, cy))
    
    # Size: cover the full overlap region + some padding
    overlap_h = int(abs(overlap_max_lat - overlap_min_lat) / (isro['max_lat'] - isro['min_lat']) * isro_h)
    overlap_w = int(abs(overlap_max_lon - overlap_min_lon) / (isro['max_lon'] - isro['min_lon']) * isro_w)
    size = max(overlap_h, overlap_w, 3000)
    size = min(size, 8000)  # cap to avoid OOM
    
    print(f'{img_path}|{lbl}|{cx}|{cy}|{size}')
")

if [ -z "$OVERLAP_LIST" ]; then
    echo "No overlapping JAXA images found!"
    exit 1
fi

total=$(echo "$OVERLAP_LIST" | wc -l)
count=0

echo "Found $total overlapping JAXA images."
echo ""

while IFS='|' read -r jaxa_img_file jaxa_lbl_file CX CY SIZE; do
    count=$((count + 1))
    
    jaxa_basename=$(basename "$jaxa_img_file" .img)
    
    echo "========================================================"
    echo " [$count/$total] $jaxa_basename"
    echo "   Searching ISRO at CX=$CX, CY=$CY, SIZE=$SIZE"
    echo "========================================================"
    
    python main.py register \
        --isro-img "$ISRO_IMG" \
        --isro-xml "$ISRO_XML" \
        --jaxa-img "$jaxa_img_file" \
        --jaxa-lbl "$jaxa_lbl_file" \
        --cx "$CX" \
        --cy "$CY" \
        --size "$SIZE" \
        --method "disk" \
        --output "$OUTPUT_DIR/match_${jaxa_basename}.png"
    
    echo ""
done <<< "$OVERLAP_LIST"

echo "Finished searching all $total overlapping JAXA images!"
