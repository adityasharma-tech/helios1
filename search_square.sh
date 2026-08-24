#!/bin/bash

# Activate virtualenv
source .venv/bin/activate

ISRO_IMG="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.img"
ISRO_XML="/media/friday/Toshiba Drive/FILES/helios/ISRO/pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20260103/data/calibrated/20260103/ch2_ohr_ncp_20260103T0410224157_d_img_d18.xml"

SQUARES_DIR="/media/friday/Toshiba Drive/FILES/helios/ISRO/mosaics/100_squares"
SIZE=1000

echo "Searching through all square images in $SQUARES_DIR ..."

for square_file in "$SQUARES_DIR"/isro_square_*.png; do
    if [ -f "$square_file" ]; then
        # Extract row and col from filename: isro_square_RRR_CCC.png
        filename=$(basename "$square_file")
        
        # Parse using Bash parameter expansion
        temp=${filename#isro_square_}
        temp=${temp%.png}
        
        # Split by underscore
        IFS='_' read -r row_str col_str <<< "$temp"
        
        # Remove leading zeros to avoid octal interpretation errors
        row=$((10#$row_str))
        col=$((10#$col_str))
        
        # Calculate center coordinates of where this square originally came from
        CX=$((col * SIZE + SIZE / 2))
        CY=$((row * SIZE + SIZE / 2))
        
        echo "--------------------------------------------------------"
        echo "Processing $filename -> Querying ISRO region centered at ($CX, $CY)"
        echo "--------------------------------------------------------"
        
        # We search the 1000x1000 query inside a 3000x3000 target region extracted from ISRO
        TARGET_SEARCH_SIZE=3000
        
        python main.py register \
            --isro-img "$ISRO_IMG" \
            --isro-xml "$ISRO_XML" \
            --query-img "$square_file" \
            --cx "$CX" \
            --cy "$CY" \
            --size "$TARGET_SEARCH_SIZE" \
            --method "disk" \
            --output "$SQUARES_DIR/match_result_${row_str}_${col_str}.png"
    fi
done

echo "Finished searching all squares!"
