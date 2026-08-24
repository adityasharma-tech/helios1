#!/bin/bash

# Define paths
ISRO_IMG="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260813/data/calibrated/20260813/ch2_tmc_ncn_20260813T1023298745_d_img_d18.img"
ISRO_XML="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/tmc_collection/data/calibrated/20260813/data/calibrated/20260813/ch2_tmc_ncn_20260813T1023298745_d_img_d18.xml"
JAXA_IMG="data/selene_overlapping_data/20081224/DTMTCO_03_05395S139E1407SC_img.img"
JAXA_LBL="data/selene_overlapping_data/20081224/DTMTCO_03_05395S139E1407SC_img.lbl"

# Setup output directory
mkdir -p results
CSV_FILE="results/batch_results.csv"

# Initialize CSV Headers
echo "cx,cy,matches,inliers,inlier_ratio,rmse,psnr,ssim" > "$CSV_FILE"

# The ISRO image is exactly 4000 px wide, so setting cx=2000 captures the entire width of the strip!
# Changing cx horizontally would just yield the exact same image patch.
# Therefore, we only need to sweep `cy` vertically across the 147,741 pixel height.
cx=2000

# Step by 1500 pixels across the strip to get exactly 97 iterations (~100 times)
for cy in $(seq 2500 1500 147000); do
    echo "======================================"
    echo "Running matching for cx=$cx, cy=$cy..."
    
    OUTPUT_IMG="results/match_${cx}_${cy}.png"
    
    # Run script (NO_COLOR=1 prevents Rich from outputting ANSI terminal color codes which break regex)
    output=$(NO_COLOR=1 .venv/bin/python main.py register \
        --isro-img "$ISRO_IMG" \
        --isro-xml "$ISRO_XML" \
        --jaxa-img "$JAXA_IMG" \
        --jaxa-lbl "$JAXA_LBL" \
        --cx $cx --cy $cy --size 5000 --max-dim 1024 \
        --method disk \
        --output "$OUTPUT_IMG" 2>&1)
        
    # Extract metrics robustly using regex
    matches=$(echo "$output" | grep "potential matches" | grep -oE "[0-9]+" | head -1)
    inliers=$(echo "$output" | grep "RANSAC Inliers:" | grep -oE "[0-9]+" | head -1)
    ratio=$(echo "$output" | grep "RANSAC Inliers:" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    
    rmse=$(echo "$output" | grep "RMSE (Accuracy)" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    psnr=$(echo "$output" | grep "PSNR (Radiometric)" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    ssim=$(echo "$output" | grep "SSIM (Structural)" | grep -oE "[0-9]+\.[0-9]+" | head -1)
    
    # Fallback defaults for failed matching regions
    if [ -z "$matches" ]; then matches="0"; fi
    if [ -z "$inliers" ]; then inliers="0"; fi
    if [ -z "$ratio" ]; then ratio="0"; fi
    if [ -z "$rmse" ]; then rmse="NaN"; fi
    if [ -z "$psnr" ]; then psnr="NaN"; fi
    if [ -z "$ssim" ]; then ssim="NaN"; fi
    
    echo "$cx,$cy,$matches,$inliers,$ratio,$rmse,$psnr,$ssim" >> "$CSV_FILE"
    echo "Finished cy=$cy | Inliers: $inliers | RMSE: $rmse"
    
    # Remove image if matching failed to save disk space and keep the results folder clean
    if [ "$inliers" -lt 5 ]; then
        rm -f "$OUTPUT_IMG"
    fi
done

echo "======================================"
echo "Batch processing complete! Top 10 best matches (sorted by inlier count):"
sort -t, -k4 -nr "$CSV_FILE" | head -n 11
