#!/bin/bash

# Define paths for the newly downloaded OHRC data
ISRO_IMG="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.img"
ISRO_XML="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"

# Using a strictly overlapping JAXA dataset found for this OHRC strip
JAXA_IMG="data/selene_overlapping_data/20080112/DTMTCO_03_01166S843E0258PS_img.img"
JAXA_LBL="data/selene_overlapping_data/20080112/DTMTCO_03_01166S843E0258PS_img.lbl"

# Setup output directory
mkdir -p results_ohrc
CSV_FILE="results_ohrc/batch_results.csv"

# Initialize CSV Headers
echo "cx,cy,matches,inliers,inlier_ratio,rmse,psnr,ssim" > "$CSV_FILE"

# The OHRC image width might be larger (e.g. 12000 pixels). 
# We'll set cx to 6000 as a rough center.
cx=6000

# Sweep across the height
for cy in $(seq 5000 5000 200000); do
    echo "======================================"
    echo "Running matching for cx=$cx, cy=$cy..."
    
    OUTPUT_IMG="results_ohrc/match_${cx}_${cy}.png"
    
    output=$(NO_COLOR=1 .venv/bin/python main.py register \
        --isro-img "$ISRO_IMG" \
        --isro-xml "$ISRO_XML" \
        --jaxa-img "$JAXA_IMG" \
        --jaxa-lbl "$JAXA_LBL" \
        --cx $cx --cy $cy --size 5000 --max-dim 1024 \
        --method disk \
        --output "$OUTPUT_IMG" 2>&1)
        
    matches=$(echo "$output" | grep "potential matches" | grep -oE "[0-9]+" | head -1)
    inliers=$(echo "$output" | grep "RANSAC Inliers:" | grep -oE "[0-9]+" | head -1)
    ratio=$(echo "$output" | grep "RANSAC Inliers:" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    rmse=$(echo "$output" | grep "RMSE (Accuracy)" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    psnr=$(echo "$output" | grep "PSNR (Radiometric)" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    ssim=$(echo "$output" | grep "SSIM (Structural)" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    
    if [ -z "$matches" ]; then matches="0"; fi
    if [ -z "$inliers" ]; then inliers="0"; fi
    if [ -z "$ratio" ]; then ratio="0"; fi
    if [ -z "$rmse" ]; then rmse="NaN"; fi
    if [ -z "$psnr" ]; then psnr="NaN"; fi
    if [ -z "$ssim" ]; then ssim="NaN"; fi
    
    echo "$cx,$cy,$matches,$inliers,$ratio,$rmse,$psnr,$ssim" >> "$CSV_FILE"
    echo "Finished cy=$cy | Inliers: $inliers | RMSE: $rmse"
    
    if [ "$inliers" -lt 5 ]; then
        rm -f "$OUTPUT_IMG"
    fi
done

echo "======================================"
echo "Batch processing complete! Top 10 best matches (sorted by inlier count):"
sort -t, -k4 -nr "$CSV_FILE" | head -n 11
